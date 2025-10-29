// Teensy 4.0 + ICM42688 direct USB (binary stream), 500 Hz
#include <SPI.h>
#include <ICM42688.h>

// --- Chip Selects (match your wiring) ---
#define INT_CS_PIN 7   // internal sensor (not used) -> keep deselected
#define EXT_CS_PIN 8   // external sensor (used)

// IMU instance uses the external CS
ICM42688 imu(SPI, EXT_CS_PIN);

// Packet: t_us + 6 floats + seq = 32 bytes
struct __attribute__((packed)) ImuPacket {
  uint32_t t_us;
  float ax, ay, az;
  float gx, gy, gz;
  uint32_t seq;
};

// --- Rates ---
static const uint32_t SAMPLE_US = 2000;  // 500 Hz pacing
// ICM-42688 ODR encodings (datasheet: *_CONFIG0, ODR bits)
#define ODR_1000HZ 0x05
#define ODR_500HZ  0x06
#define REG_GYRO_CONFIG0  0x4F
#define REG_ACCEL_CONFIG0  0x50

uint32_t next_tick = 0;
uint32_t seq = 0;

static bool imu_hw_init_500hz() {
  // Ensure both CS lines are defined and deselected BEFORE touching SPI/IMU
  pinMode(EXT_CS_PIN, OUTPUT); digitalWrite(EXT_CS_PIN, HIGH);
  pinMode(INT_CS_PIN, OUTPUT); digitalWrite(INT_CS_PIN, HIGH); // <- keep the other device inactive

  SPI.begin();
  delay(10); // small settle like your working sketch

  if (!imu.begin()) return false;

  // Match your style: set ODR explicitly (500 Hz)
  bool ok1 = imu.writeUserRegister(REG_GYRO_CONFIG0,  ODR_500HZ);
  bool ok2 = imu.writeUserRegister(REG_ACCEL_CONFIG0, ODR_500HZ);
  delay(10);
  return ok1 && ok2;
}

inline bool imu_read(float &ax, float &ay, float &az, float &gx, float &gy, float &gz) {
  // Exactly like your reference: getAGT() then acc*/gyr*()
  if (imu.getAGT()) {
    ax = imu.accX(); ay = imu.accY(); az = imu.accZ();
    gx = imu.gyrX(); gy = imu.gyrY(); gz = imu.gyrZ();
    return true;
  }
  return false;
}

void setup() {
  // High baud for binary stream
  Serial.begin(2000000);
  // Don’t print anything; Python reads raw bytes
  imu_hw_init_500hz();
  next_tick = micros();
}

void loop() {
  const uint32_t now = micros();
  if ((int32_t)(now - next_tick) >= 0) {
    next_tick += SAMPLE_US;

    ImuPacket pkt;
    pkt.t_us = now;
    pkt.seq  = seq++;

    float ax, ay, az, gx, gy, gz;
    if (!imu_read(ax, ay, az, gx, gy, gz)) {
      ax = ay = az = gx = gy = gz = 0.0f;
    }
    pkt.ax = ax; pkt.ay = ay; pkt.az = az;
    pkt.gx = gx; pkt.gy = gy; pkt.gz = gz;

    // 32-byte binary frame to PC
    Serial.write(reinterpret_cast<uint8_t*>(&pkt), sizeof(pkt));
  }
}
