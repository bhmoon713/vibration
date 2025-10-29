// Teensy 4.0 + ICM42688 direct-to-PC binary streaming @ 500 Hz
#include <SPI.h>
#include <ICM42688.h>

// ---------- Pins (edit if needed) ----------
#define IMU_CS_PIN   8

// ---------- IMU ----------
ICM42688 imu(SPI, IMU_CS_PIN);

//  t_us + 6 floats + seq = 4 + 24 + 4 = 32 bytes
struct __attribute__((packed)) ImuPacket {
  uint32_t t_us;
  float ax, ay, az;
  float gx, gy, gz;
  uint32_t seq;
};

const uint32_t SAMPLE_US = 2000;  // 500 Hz pacing
uint32_t next_tick = 0;
uint32_t seq = 0;

bool imu_init() {
  SPI.begin();
  pinMode(IMU_CS_PIN, OUTPUT);
  digitalWrite(IMU_CS_PIN, HIGH);
  if (!imu.begin()) return false;

  // Set ODR = 500 Hz (datasheet: GYRO_CONFIG0/ACCEL_CONFIG0, ODR bits = 0x06)
  imu.writeUserRegister(0x4F, 0x06); // GYRO_CONFIG0
  imu.writeUserRegister(0x50, 0x06); // ACCEL_CONFIG0
  delay(10);
  return true;
}

inline bool imu_read(float &ax, float &ay, float &az, float &gx, float &gy, float &gz) {
  if (imu.getAGT()) { // refresh Accel/Gyro/Temp
    ax = imu.accX(); ay = imu.accY(); az = imu.accZ();
    gx = imu.gyrX(); gy = imu.gyrY(); gz = imu.gyrZ();
    return true;
  }
  return false;
}

void setup() {
  // High baud to give plenty of headroom; binary script uses 2,000,000 by default
  Serial.begin(2000000);
  imu_init();
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

    // Send fixed-size binary frame
    Serial.write((uint8_t*)&pkt, sizeof(pkt));
  }
}
