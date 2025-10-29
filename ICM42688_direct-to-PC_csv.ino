// Teensy 4.0 + ICM42688 direct USB (CSV), 500 Hz, Accel ±2 g
#include <SPI.h>
#include <ICM42688.h>

#define INT_CS_PIN 7
#define EXT_CS_PIN 8

ICM42688 imu(SPI, EXT_CS_PIN);

static const uint32_t SAMPLE_US = 2000;  // 500 Hz

// ---- ICM-42688 config ----
#define REG_GYRO_CONFIG0    0x4F
#define REG_ACCEL_CONFIG0   0x50

// ODR codes (lower 4 bits)
#define ODR_1000HZ          0x05
#define ODR_500HZ           0x06

// FS bits (bits 7..5)
#define ACC_FS_2G           (0b011 << 5)  // ±2 g
#define GYR_FS_2000DPS      (0b011 << 5)  // ±2000 dps (keep default)

// Combined values we’ll write:
//   ACCEL_CONFIG0 = ACC_FS_2G | ODR_500HZ  -> 0x66
//   GYRO_CONFIG0  = GYR_FS_2000DPS | ODR_500HZ
uint32_t next_tick = 0;
uint32_t seq = 0;

static bool imu_hw_init_500hz() {
  // Ensure both devices are deselected before SPI/IMU init
  pinMode(EXT_CS_PIN, OUTPUT); digitalWrite(EXT_CS_PIN, HIGH);
  pinMode(INT_CS_PIN, OUTPUT); digitalWrite(INT_CS_PIN, HIGH);

  SPI.begin();
  delay(10);

  if (!imu.begin()) return false;

  // Set gyro: ±2000 dps @ 500 Hz
  bool ok1 = imu.writeUserRegister(REG_GYRO_CONFIG0,  GYR_FS_2000DPS | ODR_500HZ);
  // Set accel: ±2 g @ 500 Hz
  bool ok2 = imu.writeUserRegister(REG_ACCEL_CONFIG0, ACC_FS_2G      | ODR_500HZ);

  delay(10);
  return ok1 && ok2;
}

inline bool imu_read(float &ax, float &ay, float &az, float &gx, float &gy, float &gz) {
  if (imu.getAGT()) {
    // Raw readings already scaled assuming ±16 g → divide to match actual ±2 g FS
    const float ACC_SCALE_CORR = 1.0f / 8.0f;  // because 16 g / 2 g = 8
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
  Serial.begin(2000000);            // high baud to avoid bottleneck at 500 Hz
  (void)imu_hw_init_500hz();        // silent init; CSV reader expects only data/header

  // CSV header (comment out if your desktop script skips headers)
  Serial.println("t_us,ax,ay,az,gx,gy,gz,seq");

  next_tick = micros();
}

void loop() {
  const uint32_t now = micros();
  if ((int32_t)(now - next_tick) >= 0) {
    next_tick += SAMPLE_US;

    float ax, ay, az, gx, gy, gz;
    if (!imu_read(ax, ay, az, gx, gy, gz)) {
      ax = ay = az = gx = gy = gz = 0.0f;
    }

    // Ultra-lean CSV (matches your previous desktop scripts)
    Serial.print(now); Serial.print(',');
    Serial.print(ax, 6); Serial.print(',');
    Serial.print(ay, 6); Serial.print(',');
    Serial.print(az, 6); Serial.print(',');
    Serial.print(gx, 6); Serial.print(',');
    Serial.print(gy, 6); Serial.print(',');
    Serial.print(gz, 6); Serial.print(',');
    Serial.println(seq++);
  }
}
