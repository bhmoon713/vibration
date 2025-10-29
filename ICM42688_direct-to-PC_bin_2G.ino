// Teensy 4.0 + ICM42688 direct USB (binary), 500 Hz, Accel ±2 g
#include <SPI.h>
#include <ICM42688.h>

// Chip select pins
#define INT_CS_PIN 7
#define EXT_CS_PIN 8

ICM42688 imu(SPI, EXT_CS_PIN);

// 500 Hz = 2000 µs interval
static const uint32_t SAMPLE_US = 2000;

// ICM-42688 register addresses
#define REG_GYRO_CONFIG0   0x4F
#define REG_ACCEL_CONFIG0  0x50

// ODR codes
#define ODR_1000HZ         0x05
#define ODR_500HZ          0x06

// FS bits (bits 7..5)
#define ACC_FS_2G          (0b011 << 5)  // ±2 g
#define GYR_FS_2000DPS     (0b011 << 5)  // ±2000 dps

// Packet: t_us + 6 floats + seq = 32 bytes total
struct __attribute__((packed)) ImuPacket {
  uint32_t t_us;
  float ax, ay, az;
  float gx, gy, gz;
  uint32_t seq;
};

uint32_t next_tick = 0;
uint32_t seq = 0;

static bool imu_hw_init_500hz() {
  // Ensure both devices deselected before SPI init
  pinMode(EXT_CS_PIN, OUTPUT); digitalWrite(EXT_CS_PIN, HIGH);
  pinMode(INT_CS_PIN, OUTPUT); digitalWrite(INT_CS_PIN, HIGH);

  SPI.begin();
  delay(10);

  if (!imu.begin()) return false;

  // Gyro ±2000 dps, 500 Hz
  bool ok1 = imu.writeUserRegister(REG_GYRO_CONFIG0, GYR_FS_2000DPS | ODR_500HZ);
  // Accel ±2 g, 500 Hz
  bool ok2 = imu.writeUserRegister(REG_ACCEL_CONFIG0, ACC_FS_2G | ODR_500HZ);

  delay(10);
  return ok1 && ok2;
}

// Read IMU and rescale accel to match ±2 g range
inline bool imu_read(float &ax, float &ay, float &az,
                     float &gx, float &gy, float &gz) {
  if (imu.getAGT()) {
    // Library still assumes default ±16 g → divide by 8 to correct to ±2 g
    const float ACC_SCALE_CORR = 1.0f / 8.0f;
    ax = imu.accX() * ACC_SCALE_CORR;
    ay = imu.accY() * ACC_SCALE_CORR;
    az = imu.accZ() * ACC_SCALE_CORR;
    gx = imu.gyrX();
    gy = imu.gyrY();
    gz = imu.gyrZ();
    return true;
  }
  return false;
}

void setup() {
  Serial.begin(2000000);   // high-speed binary link to PC
  (void)imu_hw_init_500hz();
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

    // Send 32-byte binary frame
    Serial.write(reinterpret_cast<uint8_t*>(&pkt), sizeof(pkt));
  }
}
