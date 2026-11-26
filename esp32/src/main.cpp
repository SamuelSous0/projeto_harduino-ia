/*
 * Esteira Seletora - ESP32 (PlatformIO / src/main.cpp)
 * Integrada com Broker MQTT Público (HiveMQ / EMQX) + API FastAPI no Render + HTTP Local
 *
 * Solução Definitiva para iOS (Safari/Chrome), Android e PC na Vercel (HTTPS):
 * - Comunicação bidirecional via MQTT (ESP32 na porta TCP 1883 <-> Painel Vercel em WSS 8884)
 * - Elimina 100% os bloqueios de Mixed Content (HTTPS -> HTTP) e isolamento de IP local no iOS!
 * - Mantém também os endpoints HTTP locais (/status e /decisao) funcionando em paralelo.
 *
 * Tópicos MQTT exclusivos do projeto:
 * - Publicação de status (1s e imediato ao detectar peça): esteira-esp32-projeto/status
 * - Recebimento de decisão/comandos da IA ou painel:      esteira-esp32-projeto/comando
 *
 * Dependências no platformio.ini:
 *   lib_deps =
 *     madhephaestus/ESP32Servo
 *     bblanchon/ArduinoJson
 *     knolleary/PubSubClient
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClient.h>
#include <WiFiClientSecure.h>
#include <WebServer.h>
#include <ArduinoJson.h>
#include <ESP32Servo.h>
#include <PubSubClient.h>

// ===================== WI-FI =====================
const char* WIFI_SSID = "XtremeFibra@Ricardo&Flavia";
const char* WIFI_PASS = "22061978";

// ===================== BROKER MQTT PÚBLICO =====================
const char* MQTT_BROKER       = "broker.hivemq.com";
const int   MQTT_PORT         = 1883;
const char* MQTT_TOPIC_STATUS = "esteira-esp32-projeto/status";
const char* MQTT_TOPIC_CMD    = "esteira-esp32-projeto/comando";

// ===================== API DE IA (RENDER) =====================
const char* API_HOST = "projeto-harduino-ia.onrender.com";
const int   API_PORT = 443;

WebServer server(80);
WiFiClient espWifiClient;
PubSubClient mqttClient(espWifiClient);

// ===================== PINOS (SEGUROS NO ESP32) =====================
const int PINO_TRIG           = 5;
const int PINO_ECHO           = 18;
const int PINO_SERVO_ESTEIRA  = 23; // Servo 360° (rotação contínua)
const int PINO_SERVO_PIRAMIDE = 13; // SG90 -> y == 0 (piramide)
const int PINO_SERVO_CUBO     = 14; // SG90 -> y == 1 (cubo) [GPIO 14]

// ===================== PARÂMETROS DA ESTEIRA =====================
const float DISTANCIA_DETECCAO_CM = 15.0f;
const int   VEL_ESTEIRA_FRENTE    = 110;
const int   VEL_ESTEIRA_PARAR     = 90;

const int   ANGULO_REPOUSO        = 20;
const int   ANGULO_CHUTE          = 85;

const unsigned long INTERVALO_ULTRASSONICO_MS = 100;
const unsigned long COOLDOWN_DETECCAO_MS      = 2500;
const unsigned long TIMEOUT_DECISAO_MS        = 12000;
const unsigned long INTERVALO_MQTT_STATUS_MS  = 1000;
const unsigned long INTERVALO_MQTT_RECONN_MS  = 3000;

Servo servoEsteira;
Servo servoPiramide;
Servo servoCubo;

volatile bool pecaDetectada = false;
bool esteiraRodando = true;
float ultimaDistanciaCM = -1.0f;
int totalPecasClassificadas = 0;
String ultimaClasseStr = "nenhuma";
int ultimoY = -1;

unsigned long ultimaDeteccao = 0;
unsigned long ultimaLeituraSensor = 0;
unsigned long ultimaLeituraPrint = 0;
unsigned long ultimoEnvioMqtt = 0;
unsigned long ultimaTentativaMqtt = 0;

bool temDecisaoPendente = false;
int  decisaoPendenteY   = -1;

// Declaração antecipada para uso nos callbacks
void publicarStatusMQTT(const char* evento = "status");

// ===================== ESTEIRA E SERVOS =====================
void moverEsteira(bool ligada) {
  esteiraRodando = ligada;
  servoEsteira.write(ligada ? VEL_ESTEIRA_FRENTE : VEL_ESTEIRA_PARAR);
}

void acionarSG90(Servo &servo, int pino, int atrasoAntesMs) {
  if (atrasoAntesMs > 0) {
    delay(atrasoAntesMs);
  }
  servo.attach(pino, 544, 2400);
  servo.write(ANGULO_CHUTE);
  delay(400);
  servo.write(ANGULO_REPOUSO);
  delay(300);
  servo.detach(); // Solta PWM para evitar tremor contínuo
}

// ===================== ULTRASSÔNICO RÁPIDO =====================
float lerDistanciaUnicaCM() {
  digitalWrite(PINO_TRIG, LOW);
  delayMicroseconds(4);
  digitalWrite(PINO_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PINO_TRIG, LOW);

  long duracao = pulseIn(PINO_ECHO, HIGH, 5000);
  if (duracao <= 150) {
    return -1.0f;
  }

  float dist = (duracao * 0.0343f) / 2.0f;
  if (dist < 2.0f || dist > 85.0f) {
    return -1.0f;
  }
  return dist;
}

float lerDistanciaFiltradaCM() {
  const int AMOSTRAS = 3;
  float soma = 0.0f;
  int validas = 0;

  for (int i = 0; i < AMOSTRAS; i++) {
    float d = lerDistanciaUnicaCM();
    if (d > 0.0f) {
      soma += d;
      validas++;
    }
    delay(3);
  }
  return (validas == 0) ? -1.0f : (soma / validas);
}

// ===================== WARM-UP RENDER =====================
void acordarAPI() {
  WiFiClientSecure client;
  client.setInsecure();
  client.setTimeout(60000);

  Serial.println("[IA] Chamando /health para acordar a API...");
  if (!client.connect(API_HOST, API_PORT)) {
    Serial.println("[IA] Falha ao conectar no /health.");
    return;
  }

  client.println("GET /health HTTP/1.1");
  client.printf("Host: %s\r\n", API_HOST);
  client.println("Connection: close");
  client.println();

  unsigned long inicio = millis();
  while (client.connected() && !client.available() && (millis() - inicio < 60000)) {
    delay(50);
  }

  String resp = "";
  while (client.available()) {
    resp += client.readStringUntil('\n');
  }
  client.stop();

  int ini = resp.indexOf('{');
  int fim = resp.lastIndexOf('}');
  if (ini != -1 && fim != -1 && fim > ini) {
    JsonDocument doc;
    if (!deserializeJson(doc, resp.substring(ini, fim + 1))) {
      bool carregado = doc["modelo_carregado"] | false;
      Serial.printf("[IA] API online! modelo_carregado = %s\n", carregado ? "true" : "false");
      return;
    }
  }
  Serial.println("[IA] Resposta /health recebida.");
}

// ===================== EXECUÇÃO DA DECISÃO =====================
void executarDecisao(int y) {
  moverEsteira(true);

  switch (y) {
    case 0:
      Serial.println("[ESTEIRA] Desviando PIRAMIDE (y = 0)");
      acionarSG90(servoPiramide, PINO_SERVO_PIRAMIDE, 200);
      break;
    case 1:
      Serial.println("[ESTEIRA] Desviando CUBO (y = 1)");
      acionarSG90(servoCubo, PINO_SERVO_CUBO, 500);
      break;
    case -1:
    default:
      Serial.println("[ESTEIRA] Objeto NAO RECONHECIDO (y = -1) -> Passar reto!");
      break;
  }

  pecaDetectada = false;
  ultimaDeteccao = millis();
  publicarStatusMQTT("decisao_executada");
}

bool agendarDecisaoJSON(const String &payload) {
  JsonDocument doc;
  DeserializationError erro = deserializeJson(doc, payload);
  if (erro) {
    Serial.println("[ERRO] JSON invalido: " + payload);
    return false;
  }

  // Caso o painel tenha enviado apenas um ping de handshake {"tipo":"ping_ponte"}
  if (doc["tipo"].is<const char*>()) {
    const char* tipo = doc["tipo"];
    if (strcmp(tipo, "ping_ponte") == 0) {
      const char* origem = doc["origem"] | "Painel Web Vercel";
      Serial.printf("[MQTT] Painel Web CONECTADO via MQTT! (Origem: %s)\n", origem);
      publicarStatusMQTT("ponte_conectada");
      return true;
    }
  }

  int y = doc["y"] | -1;
  const char* classe = doc["classe"] | (y == 0 ? "piramide" : (y == 1 ? "cubo" : "nao_reconhecido"));
  const char* acao = doc["acao"] | (y == 0 ? "desviar_piramide" : (y == 1 ? "desviar_cubo" : "passar_reto"));
  float confianca = doc["confianca"] | 1.0f;

  Serial.printf("[IA/DECISAO] classe=%s | y=%d | acao=%s | confianca=%.2f%%\n",
                classe, y, acao, confianca * 100.0f);

  ultimaClasseStr = String(classe);
  ultimoY = y;
  totalPecasClassificadas++;

  decisaoPendenteY = y;
  temDecisaoPendente = true;
  return true;
}

// ===================== MQTT (PUBSUBCLIENT) =====================
String montarJsonStatus(const char* evento = "status") {
  JsonDocument doc;
  doc["peca_detectada"]      = pecaDetectada;
  doc["status_esteira"]      = pecaDetectada ? "pausada_aguardando_ia" : "rodando";
  doc["distancia_cm"]        = serialized(String(ultimaDistanciaCM, 1));
  doc["ultima_decisao"]      = ultimaClasseStr;
  doc["ultimo_y"]            = ultimoY;
  doc["pecas_classificadas"] = totalPecasClassificadas;
  doc["ip"]                  = WiFi.localIP().toString();
  doc["evento"]              = evento;

  String out;
  serializeJson(doc, out);
  return out;
}

void publicarStatusMQTT(const char* evento) {
  if (!mqttClient.connected()) return;
  String json = montarJsonStatus(evento);
  // retained = true permite que o celular/PC receba o estado atual no exato instante em que abrir o painel!
  mqttClient.publish(MQTT_TOPIC_STATUS, json.c_str(), true);
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String msg;
  msg.reserve(length + 1);
  for (unsigned int i = 0; i < length; i++) {
    msg += (char)payload[i];
  }
  msg.trim();

  Serial.printf("[MQTT] Mensagem recebida em [%s]: %s\n", topic, msg.c_str());

  // Aceita tanto JSON completo {"classe":"cubo","y":1,...} quanto "0", "1", "-1" direto
  if (msg == "0" || msg == "1" || msg == "-1") {
    int y = msg.toInt();
    String simulado = "{\"y\":" + String(y) + ",\"classe\":\"" +
                      (y == 0 ? "piramide" : (y == 1 ? "cubo" : "nao_reconhecido")) +
                      "\",\"confianca\":1.0}";
    agendarDecisaoJSON(simulado);
    return;
  }

  agendarDecisaoJSON(msg);
}

void verificarConexaoMQTT() {
  if (WiFi.status() != WL_CONNECTED) return;
  if (mqttClient.connected()) return;

  unsigned long agora = millis();
  if (agora - ultimaTentativaMqtt < INTERVALO_MQTT_RECONN_MS) return;
  ultimaTentativaMqtt = agora;

  String clientId = "ESP32-Esteira-" + String((uint32_t)ESP.getEfuseMac(), HEX);
  Serial.printf("[MQTT] Conectando ao broker %s:%d como %s...\n",
                MQTT_BROKER, MQTT_PORT, clientId.c_str());

  if (mqttClient.connect(clientId.c_str())) {
    Serial.println("[MQTT] CONECTADO ao Broker MQTT com sucesso!");
    mqttClient.subscribe(MQTT_TOPIC_CMD);
    Serial.printf("[MQTT] Inscrito em: %s\n", MQTT_TOPIC_CMD);
    Serial.printf("[MQTT] Publicando status em: %s\n", MQTT_TOPIC_STATUS);
    publicarStatusMQTT("esp32_online");
  } else {
    Serial.printf("[MQTT] Falha na conexao (rc=%d). Tentando novamente em 3s...\n",
                  mqttClient.state());
  }
}

// ===================== ENDPOINTS HTTP LOCAIS (MANTIDOS EM PARALELO) =====================
void enviarCORS() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  server.sendHeader("Access-Control-Allow-Headers", "Content-Type");
}

void handleOptions() {
  enviarCORS();
  server.send(204);
}

void handleStatus() {
  enviarCORS();
  server.send(200, "application/json", montarJsonStatus("http_status"));
}

void handleDecisao() {
  enviarCORS();

  if (server.hasArg("plain") && server.arg("plain").length() > 0) {
    if (agendarDecisaoJSON(server.arg("plain"))) {
      server.send(200, "application/json", "{\"status\":\"ok\"}");
    } else {
      server.send(400, "application/json", "{\"erro\":\"json_invalido\"}");
    }
    return;
  }

  if (server.hasArg("y")) {
    int y = server.arg("y").toInt();
    String simulado = "{\"y\":" + String(y) + ",\"classe\":\"via_http\",\"acao\":\"decisao\",\"confianca\":1.0}";
    agendarDecisaoJSON(simulado);
    server.send(200, "application/json", "{\"status\":\"ok\"}");
    return;
  }

  server.send(400, "text/plain", "Envie JSON no body ou parametro ?y=0|1|-1");
}

// ===================== SETUP E LOOP =====================
void setup() {
  Serial.begin(115200);
  delay(300);

  pinMode(PINO_TRIG, OUTPUT);
  pinMode(PINO_ECHO, INPUT);

  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  servoEsteira.attach(PINO_SERVO_ESTEIRA, 544, 2400);
  moverEsteira(true);

  acionarSG90(servoPiramide, PINO_SERVO_PIRAMIDE, 0);
  acionarSG90(servoCubo, PINO_SERVO_CUBO, 0);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Conectando ao Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.printf("\n[OK] Wi-Fi conectado! IP da ESP32: http://%s\n", WiFi.localIP().toString().c_str());

  // Configura MQTT
  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
  mqttClient.setBufferSize(512);
  mqttClient.setCallback(mqttCallback);
  verificarConexaoMQTT();

  acordarAPI();

  server.on("/status",  HTTP_OPTIONS, handleOptions);
  server.on("/decisao", HTTP_OPTIONS, handleOptions);
  server.on("/status",  HTTP_GET,     handleStatus);
  server.on("/decisao", HTTP_POST,    handleDecisao);
  server.on("/decisao", HTTP_GET,     handleDecisao);
  server.begin();

  Serial.println("[OK] WebServer HTTP + Cliente MQTT prontos!");
}

void loop() {
  // Reconexão automática de Wi-Fi se cair
  if (WiFi.status() != WL_CONNECTED) {
    WiFi.reconnect();
  }

  // Mantém conexão MQTT ativa e processa mensagens do tópico de comando
  verificarConexaoMQTT();
  mqttClient.loop();

  // Mantém WebServer HTTP local ativo em paralelo
  server.handleClient();

  if (temDecisaoPendente) {
    temDecisaoPendente = false;
    executarDecisao(decisaoPendenteY);
  }

  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "0" || cmd == "1" || cmd == "-1") {
      executarDecisao(cmd.toInt());
    }
  }

  unsigned long agora = millis();

  if (pecaDetectada && (agora - ultimaDeteccao > TIMEOUT_DECISAO_MS)) {
    Serial.println("[AVISO] Timeout esperando decisao da IA -> retomando esteira (passar reto).");
    executarDecisao(-1);
  }

  if (!pecaDetectada && (agora - ultimaLeituraSensor >= INTERVALO_ULTRASSONICO_MS)) {
    ultimaLeituraSensor = agora;
    float dist = lerDistanciaFiltradaCM();
    ultimaDistanciaCM = dist;

    if (agora - ultimaLeituraPrint >= 500) {
      ultimaLeituraPrint = agora;
      if (dist < 0.0f) {
        Serial.println("[ULTRASSONICO] Sem objeto proximo");
      } else {
        Serial.printf("[ULTRASSONICO] Distancia: %.1f cm\n", dist);
      }
    }

    if (dist > 2.0f && dist <= DISTANCIA_DETECCAO_CM && (agora - ultimaDeteccao > COOLDOWN_DETECCAO_MS)) {
      pecaDetectada = true;
      ultimaDeteccao = agora;
      moverEsteira(false);
      Serial.printf("[ULTRASSONICO] Peca detectada a %.1f cm! Esteira pausada aguardando IA...\n", dist);
      // Publica imediatamente no MQTT para o celular/PC iniciar a leitura na hora!
      publicarStatusMQTT("peca_detectada");
      ultimoEnvioMqtt = agora;
    }
  }

  // Publica o status no MQTT automaticamente a cada 1 segundo
  if (agora - ultimoEnvioMqtt >= INTERVALO_MQTT_STATUS_MS) {
    ultimoEnvioMqtt = agora;
    publicarStatusMQTT("heartbeat");
  }
}
