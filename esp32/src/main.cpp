/*
 * Esteira Seletora - ESP32 (PlatformIO / src/main.cpp)
 *
 * Funciona em QUALQUER lugar (Universidade / Casa / Sem Roteador):
 * 1. BLUETOOTH BLE NATIVO ("ESP32-Esteira-IA"):
 *    - Liga imediatamente no boot (antes mesmo do Wi-Fi!).
 *    - Conecta direto com o navegador do PC ou Celular via Web Bluetooth,
 *      SEM depender do firewall/Wi-Fi da Universidade e SEM precisar de IP!
 * 2. WI-FI HTTP LOCAL (IP da ESP32 -> GET /status e POST /decisao):
 *    - Mantém 100% o padrão original por IP (timeout de Wi-Fi de 8s para não
 *      travar o boot caso o Wi-Fi da universidade não esteja disponível).
 * 3. ULTRASSÔNICO PRECISO (45 ms + filtro de mediana) + SERVOS ORIGINAIS:
 *    - TRIG GPIO 5, ECHO GPIO 18, Esteira GPIO 23, Pirâmide GPIO 13 (200ms), Cubo GPIO 14 (500ms).
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <WebServer.h>
#include <ArduinoJson.h>
#include <ESP32Servo.h>

// Bibliotecas Bluetooth BLE nativas do core ESP32 (não precisa instalar nada extra no platformio.ini)
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ===================== WI-FI =====================
const char* WIFI_SSID = "XtremeFibra@Ricardo&Flavia";
const char* WIFI_PASS = "22061978";

// ===================== BLUETOOTH BLE (WEB BLUETOOTH) =====================
#define BLE_DEVICE_NAME       "ESP32-Esteira-IA"
#define BLE_SERVICE_UUID      "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define BLE_CHAR_STATUS_UUID  "beb5483e-36e1-4688-b7f5-ea07361b26a8" // Notify + Read (ESP32 -> Painel)
#define BLE_CHAR_DECISAO_UUID "828917c1-565d-4797-83a9-951404a3b083" // Write (Painel -> ESP32)

BLEServer* pBleServer = nullptr;
BLECharacteristic* pStatusChar = nullptr;
BLECharacteristic* pDecisaoChar = nullptr;
volatile bool bleClienteConectado = false;

// ===================== API DE IA (RENDER) =====================
const char* API_HOST = "projeto-harduino-ia.onrender.com";
const int   API_PORT = 443;

WebServer server(80);

// ===================== PINOS (SEGUROS NO ESP32) =====================
const int PINO_TRIG           = 5;
const int PINO_ECHO           = 18;
const int PINO_SERVO_ESTEIRA  = 23; // Servo 360° (rotação contínua)
const int PINO_SERVO_PIRAMIDE = 13; // SG90 -> y == 0 (piramide)
const int PINO_SERVO_CUBO     = 14; // SG90 -> y == 1 (cubo) [GPIO 14]

// ===================== PARÂMETROS DA ESTEIRA E ULTRASSÔNICO =====================
const float DISTANCIA_DETECCAO_CM = 15.0f;
const float DISTANCIA_MIN_CM      = 1.8f;
const float DISTANCIA_MAX_CM      = 120.0f;

const int   VEL_ESTEIRA_FRENTE    = 110;
const int   VEL_ESTEIRA_PARAR     = 90;

const int   ANGULO_REPOUSO        = 20;
const int   ANGULO_CHUTE          = 85;

const unsigned long INTERVALO_ULTRASSONICO_MS = 45;
const int           CONFIRMACOES_NECESSARIAS  = 2;
const unsigned long COOLDOWN_DETECCAO_MS      = 2500;
const unsigned long TIMEOUT_DECISAO_MS        = 12000;
const unsigned long INTERVALO_BLE_NOTIFY_MS   = 450;

Servo servoEsteira;
Servo servoPiramide;
Servo servoCubo;

volatile bool pecaDetectada = false;
float ultimaDistanciaCM = -1.0f;
int totalPecasClassificadas = 0;
String ultimaClasseStr = "nenhuma";
int ultimoY = -1;

float janelaDist[3] = {-1.0f, -1.0f, -1.0f};
int   indiceJanela = 0;
int   contagemPecaConsecutiva = 0;

bool ponteIpConectada = false;
unsigned long ultimoPingIp = 0;
unsigned long ultimoNotifyBle = 0;

unsigned long ultimaDeteccao = 0;
unsigned long ultimaLeituraSensor = 0;
unsigned long ultimaLeituraPrint = 0;

bool temDecisaoPendente = false;
int  decisaoPendenteY   = -1;

// Declarações antecipadas
bool agendarDecisaoJSON(const String &payload);
void notificarStatusBLE();

// ===================== ESTEIRA E SERVOS =====================
void moverEsteira(bool ligada) {
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

// ===================== ULTRASSÔNICO PRECISO =====================
void resetarFiltroUltrassonico() {
  janelaDist[0] = -1.0f;
  janelaDist[1] = -1.0f;
  janelaDist[2] = -1.0f;
  contagemPecaConsecutiva = 0;
}

float lerPulsoUltrassonicoCM() {
  digitalWrite(PINO_TRIG, LOW);
  delayMicroseconds(4);
  digitalWrite(PINO_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PINO_TRIG, LOW);

  unsigned long duracao = pulseIn(PINO_ECHO, HIGH, 12000);
  if (duracao == 0) {
    return -1.0f;
  }

  float dist = (duracao * 0.0343f) / 2.0f;
  if (dist < DISTANCIA_MIN_CM || dist > DISTANCIA_MAX_CM) {
    return -1.0f;
  }
  return dist;
}

float atualizarUltrassonicoPrecisoCM() {
  float leituraAtual = lerPulsoUltrassonicoCM();
  janelaDist[indiceJanela] = leituraAtual;
  indiceJanela = (indiceJanela + 1) % 3;

  float validas[3];
  int n = 0;
  for (int i = 0; i < 3; i++) {
    if (janelaDist[i] > 0.0f) {
      validas[n++] = janelaDist[i];
    }
  }

  if (n == 0) {
    contagemPecaConsecutiva = 0;
    return -1.0f;
  }

  for (int i = 0; i < n - 1; i++) {
    for (int j = i + 1; j < n; j++) {
      if (validas[j] < validas[i]) {
        float tmp = validas[i];
        validas[i] = validas[j];
        validas[j] = tmp;
      }
    }
  }

  float distEstavel = (n == 3) ? validas[1] : ((n == 2) ? (validas[0] + validas[1]) * 0.5f : validas[0]);

  if (leituraAtual > 0.0f && leituraAtual <= DISTANCIA_DETECCAO_CM) {
    contagemPecaConsecutiva++;
  } else if (distEstavel > DISTANCIA_DETECCAO_CM + 1.5f || leituraAtual < 0.0f) {
    contagemPecaConsecutiva = 0;
  }

  return distEstavel;
}

// ===================== JSON DE STATUS (USADO NO IP HTTP E NO BLUETOOTH) =====================
String montarJsonStatus() {
  String json = "{\"peca_detectada\":";
  json += (pecaDetectada ? "true" : "false");
  json += ",\"distancia_cm\":";
  json += String(ultimaDistanciaCM, 1);
  json += ",\"pecas_classificadas\":";
  json += String(totalPecasClassificadas);
  json += ",\"ultima_decisao\":\"";
  json += ultimaClasseStr;
  json += "\",\"ultimo_y\":";
  json += String(ultimoY);
  json += "}";
  return json;
}

// ===================== CALLBACKS BLUETOOTH BLE =====================
class ServerBleCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) override {
    bleClienteConectado = true;
    Serial.println("[BLUETOOTH] Painel Web CONECTADO via Bluetooth BLE!");
  }
  void onDisconnect(BLEServer* pServer) override {
    bleClienteConectado = false;
    Serial.println("[BLUETOOTH] Painel Web desconectado do Bluetooth. Reiniciando anuncio BLE...");
    BLEDevice::startAdvertising();
  }
};

class DecisaoBleCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic* pCharacteristic) override {
    std::string valor = pCharacteristic->getValue();
    if (valor.length() > 0) {
      String msg = String(valor.c_str());
      msg.trim();
      Serial.printf("[BLUETOOTH] Decisao recebida via BLE: %s\n", msg.c_str());

      if (msg == "0" || msg == "1" || msg == "-1") {
        int y = msg.toInt();
        String simulado = "{\"y\":" + String(y) + ",\"classe\":\"" +
                          (y == 0 ? "piramide" : (y == 1 ? "cubo" : "nao_reconhecido")) +
                          "\",\"confianca\":1.0}";
        agendarDecisaoJSON(simulado);
      } else {
        agendarDecisaoJSON(msg);
      }
    }
  }
};

void iniciarBluetoothBLE() {
  BLEDevice::init(BLE_DEVICE_NAME);
  pBleServer = BLEDevice::createServer();
  pBleServer->setCallbacks(new ServerBleCallbacks());

  BLEService* pService = pBleServer->createService(BLE_SERVICE_UUID);

  // Característica de Status (ESP32 -> Navegador via Notify e Read)
  pStatusChar = pService->createCharacteristic(
    BLE_CHAR_STATUS_UUID,
    BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
  );
  pStatusChar->addDescriptor(new BLE2902());

  // Característica de Decisão (Navegador -> ESP32 via Write)
  pDecisaoChar = pService->createCharacteristic(
    BLE_CHAR_DECISAO_UUID,
    BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR
  );
  pDecisaoChar->setCallbacks(new DecisaoBleCallbacks());

  pService->start();

  BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(BLE_SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);
  pAdvertising->setMinPreferred(0x12);
  BLEDevice::startAdvertising();

  Serial.printf("[OK] Bluetooth BLE ativo! Nome: %s (Conecta direto sem precisar de Wi-Fi/IP)\n", BLE_DEVICE_NAME);
}

void notificarStatusBLE() {
  if (!pStatusChar) return;
  String json = montarJsonStatus();
  pStatusChar->setValue(json.c_str());
  if (bleClienteConectado) {
    pStatusChar->notify();
  }
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
  resetarFiltroUltrassonico();
  ultimaDeteccao = millis();
  notificarStatusBLE();
}

bool agendarDecisaoJSON(const String &payload) {
  JsonDocument doc;
  DeserializationError erro = deserializeJson(doc, payload);
  if (erro) {
    Serial.println("[ERRO] JSON invalido: " + payload);
    return false;
  }

  int y = doc["y"] | -1;
  const char* classe = doc["classe"] | "nao_reconhecido";
  const char* acao = doc["acao"] | "passar_reto";
  float confianca = doc["confianca"] | 0.0f;

  Serial.printf("[IA] classe=%s | y=%d | acao=%s | confianca=%.2f%%\n",
                classe, y, acao, confianca * 100.0f);

  ultimaClasseStr = String(classe);
  ultimoY = y;
  totalPecasClassificadas++;

  decisaoPendenteY = y;
  temDecisaoPendente = true;
  return true;
}

// ===================== ENDPOINTS HTTP POR IP (COM CORS) =====================
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
  unsigned long agora = millis();
  if (!ponteIpConectada || server.hasArg("connect") || (agora - ultimoPingIp > 5000)) {
    ponteIpConectada = true;
    Serial.printf("[PONTE IP] Painel conectado via IP! (Cliente: %s)\n",
                  server.client().remoteIP().toString().c_str());
  }
  ultimoPingIp = agora;

  server.send(200, "application/json", montarJsonStatus());
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
  digitalWrite(PINO_TRIG, LOW);

  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  servoEsteira.attach(PINO_SERVO_ESTEIRA, 544, 2400);
  moverEsteira(true);

  acionarSG90(servoPiramide, PINO_SERVO_PIRAMIDE, 0);
  acionarSG90(servoCubo, PINO_SERVO_CUBO, 0);

  // 1. Inicia o Bluetooth BLE imediatamente (funciona na universidade mesmo sem Wi-Fi!)
  iniciarBluetoothBLE();

  // 2. Tenta conectar ao Wi-Fi por até 8 segundos (para modo IP caso esteja disponível)
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.printf("Tentando Wi-Fi '%s' (timeout 8s para nao travar se estiver na Universidade)", WIFI_SSID);
  unsigned long inicioWifi = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - inicioWifi < 8000)) {
    delay(400);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[OK] Wi-Fi conectado! IP da ESP32: http://%s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\n[AVISO] Wi-Fi nao encontrado. Rodando 100% via BLUETOOTH BLE ('ESP32-Esteira-IA')!");
  }

  server.on("/status",  HTTP_OPTIONS, handleOptions);
  server.on("/decisao", HTTP_OPTIONS, handleOptions);
  server.on("/status",  HTTP_GET,     handleStatus);
  server.on("/decisao", HTTP_POST,    handleDecisao);
  server.on("/decisao", HTTP_GET,     handleDecisao);
  server.begin();

  Serial.println("[OK] Sistema pronto! Pode conectar por Bluetooth ('ESP32-Esteira-IA') ou pelo IP.");
}

void loop() {
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
    float dist = atualizarUltrassonicoPrecisoCM();
    ultimaDistanciaCM = dist;

    if (agora - ultimaLeituraPrint >= 450) {
      ultimaLeituraPrint = agora;
      if (dist < 0.0f) {
        Serial.println("[ULTRASSONICO] Sem objeto proximo");
      } else {
        Serial.printf("[ULTRASSONICO] Distancia: %.1f cm (conf=%d/%d)\n",
                      dist, contagemPecaConsecutiva, CONFIRMACOES_NECESSARIAS);
      }
    }

    if (contagemPecaConsecutiva >= CONFIRMACOES_NECESSARIAS &&
        dist > DISTANCIA_MIN_CM &&
        dist <= DISTANCIA_DETECCAO_CM &&
        (agora - ultimaDeteccao > COOLDOWN_DETECCAO_MS)) {
      pecaDetectada = true;
      ultimaDeteccao = agora;
      moverEsteira(false);
      Serial.printf("[ULTRASSONICO] Peca detectada a %.1f cm! Esteira pausada aguardando IA...\n", dist);
      // Notifica imediatamente via Bluetooth BLE se estiver conectado!
      notificarStatusBLE();
      ultimoNotifyBle = agora;
    }
  }

  // Envia atualização periódica pelo Bluetooth BLE
  if (bleClienteConectado && (agora - ultimoNotifyBle >= INTERVALO_BLE_NOTIFY_MS)) {
    ultimoNotifyBle = agora;
    notificarStatusBLE();
  }
}
