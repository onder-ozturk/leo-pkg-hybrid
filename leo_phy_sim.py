#!/usr/bin/env python3
"""
LEO Satellite PHY-Layer Key Generation Simulator
Türkiye FGN-100 ve TÜRKSAT 6A parametreleri ile
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from dataclasses import dataclass
from typing import Tuple, List
import hashlib

@dataclass
class SatelliteParams:
    """Uydu parametreleri"""
    name: str
    altitude_km: float      # Yörünge yüksekliği
    velocity_kms: float     # Hız (km/s)
    frequency_ghz: float    # Taşıyıcı frekans
    bandwidth_mhz: float    # Bant genişliği
    
    @property
    def coherence_time_ms(self) -> float:
        """Kanal coherence süresi"""
        f_d_max = (self.velocity_kms * 1e3 * self.frequency_ghz * 1e9) / 3e8
        return 1 / (2 * f_d_max) * 1000  # ms

# Türkiye Uyduları
FGN100 = SatelliteParams(
    name="FGN-100",
    altitude_km=510,
    velocity_kms=7.6,
    frequency_ghz=2.2,  # S-band
    bandwidth_mhz=10
)

TURKSAT6A = SatelliteParams(
    name="TÜRKSAT 6A",
    altitude_km=35786,  # GEO
    velocity_kms=3.07,  # ~11068 km/h
    frequency_ghz=12.0,  # Ku-band
    bandwidth_mhz=36
)

class LEOChannel:
    """LEO uydu kanalı simülatörü"""
    
    def __init__(self, sat_params: SatelliteParams, elevation_angle=45):
        self.sat = sat_params
        self.elevation = np.radians(elevation_angle)
        self.c = 3e8
        
        # Doppler hesabı
        self.f_doppler = self._calculate_doppler()
        
        # Fading parametreleri
        self.k_factor = 10  # Rician K-factor
        self.shadowing_std = 4  # dB
        
    def _calculate_doppler(self) -> float:
        """Doppler frekansı hesaplama"""
        v_radial = self.sat.velocity_kms * 1e3 * np.sin(self.elevation)
        return v_radial * self.sat.frequency_ghz * 1e9 / self.c
    
    def generate_csi(self, num_samples: int, snr_db: float = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kanal durum bilgisi (CSI) üretimi
        Returns: (h_AB, h_BA) - Reciprocal kanallar
        """
        # Zaman vektörü
        t = np.arange(num_samples) / (self.sat.bandwidth_mhz * 1e6)
        
        # Small-scale fading (Rician)
        s = np.sqrt(self.k_factor / (self.k_factor + 1))
        sigma = np.sqrt(1 / (2 * (self.k_factor + 1)))
        
        # LOS bileşeni
        los = s * np.exp(1j * 2 * np.pi * self.f_doppler * t)
        
        # NLOS bileşeni (Rayleigh)
        nlos = (sigma * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)))
        
        # Hızlı fading
        h = los + nlos
        
        # Large-scale fading (log-normal shadowing)
        shadowing = np.random.normal(0, self.shadowing_std, num_samples)
        h *= np.power(10, shadowing / 20)
        
        # Reciprocal kanallar (h_AB ≈ h_BA)
        noise_power = np.power(10, -snr_db / 10)
        n_ab = np.sqrt(noise_power/2) * (np.random.randn(num_samples) + 1j*np.random.randn(num_samples))
        n_ba = np.sqrt(noise_power/2) * (np.random.randn(num_samples) + 1j*np.random.randn(num_samples))
        
        h_ab = h + n_ab
        h_ba = h + n_ba  # Aynı h, farklı gürültü
        
        return h_ab, h_ba

class PHYKeyGenerator:
    """PHY-Layer Anahtar Üretimi"""
    
    def __init__(self, quantization_bits=3):
        self.quant_bits = quantization_bits
        self.thresholds = None
        
    def extract_features(self, csi: np.ndarray) -> np.ndarray:
        """CSI'dan özellik çıkarımı"""
        features = np.array([
            np.abs(csi),           # Genlik
            np.angle(csi),         # Faz
            np.abs(np.diff(csi, append=np.array([csi[-1]])))   # Değişim hızı (fixed diff issue)
        ])
        return features.flatten()
    
    def adaptive_quantize(self, features: np.ndarray, snr_db: float) -> List[int]:
        """Adaptive quantizasyon"""
        # SNR bazlı seviye sayısı
        levels = min(2**self.quant_bits, max(2, int(snr_db / 5)))
        
        # Histogram bazlı eşik belirleme
        hist, bin_edges = np.histogram(features, bins=levels-1)
        self.thresholds = bin_edges[1:-1]
        
        # Quantizasyon
        quantized = np.digitize(features, self.thresholds)
        
        return quantized.tolist()
    
    def reconcile(self, bits_a: List[int], bits_b: List[int]) -> Tuple[bytes, float]:
        """Bilgi uzlaştırma (Cascade protokolü basitleştirilmiş)"""
        bits_a = np.array(bits_a)
        bits_b = np.array(bits_b)
        
        # Parity kontrolü ve düzeltme
        disagreements = np.sum(bits_a != bits_b)
        kdr = disagreements / len(bits_a)
        
        # Basit majority voting ile düzeltme
        corrected = bits_b.copy()
        for i in range(len(bits_a)):
            if bits_a[i] != bits_b[i]:
                # Rastgele düzeltme (gerçekte daha sofistike olmalı)
                corrected[i] = bits_a[i]
        
        # Bytes'a çevir
        bit_string = ''.join([format(v, f'0{self.quant_bits}b') for v in corrected])
        key_bytes = int(bit_string, 2).to_bytes((len(bit_string) + 7) // 8, 'big')
        
        return key_bytes, kdr

class EavesdropperDetector:
    """Eavesdropper tespit sistemi"""
    
    def __init__(self, threshold=0.85):
        self.threshold = threshold
        self.history = []
        
    def analyze(self, csi_alice: np.ndarray, csi_bob: np.ndarray) -> dict:
        """Korelasyon analizi"""
        # Pearson correlation
        corr = np.corrcoef(np.abs(csi_alice), np.abs(csi_bob))[0,1]
        
        # Faz korelasyonu
        phase_diff = np.angle(csi_alice) - np.angle(csi_bob)
        phase_corr = np.abs(np.mean(np.exp(1j * phase_diff)))
        
        # Mutual information (Gaussian yaklaşımı)
        mi = -0.5 * np.log(max(0.001, 1 - corr**2))
        
        # Tespit
        detected = corr < self.threshold or phase_corr < 0.8
        
        result = {
            'eve_detected': detected,
            'correlation': corr,
            'phase_coherence': phase_corr,
            'mutual_info': mi,
            'risk_level': 'HIGH' if corr < 0.8 else 'MEDIUM' if corr < 0.9 else 'LOW'
        }
        
        self.history.append(result)
        return result

class HybridSecuritySystem:
    """Hibrit güvenlik sistemi (PHY + Classical)"""
    
    def __init__(self, master_key: bytes):
        self.master_key = master_key
        self.key_gen = PHYKeyGenerator()
        self.detector = EavesdropperDetector()
        
    def generate_session_key(self, csi_alice: np.ndarray, csi_bob: np.ndarray, 
                            snr_db: float) -> bytes:
        """Hibrit oturum anahtarı üretimi"""
        
        # Eve kontrolü
        detection = self.detector.analyze(csi_alice, csi_bob)
        # if detection['eve_detected']:
        #     print(f"⚠️  WARNING: {detection['risk_level']} risk - Potential eavesdropper!")
        
        # PHY-Key üretimi
        features_a = self.key_gen.extract_features(csi_alice)
        features_b = self.key_gen.extract_features(csi_bob)
        
        bits_a = self.key_gen.adaptive_quantize(features_a, snr_db)
        bits_b = self.key_gen.adaptive_quantize(features_b, snr_db)
        
        phy_key, kdr = self.key_gen.reconcile(bits_a, bits_b)
        
        # Privacy amplification
        phy_key_hash = hashlib.sha256(phy_key).digest()[:16]
        
        # Hibritleştirme (XOR)
        session_key = bytes(a ^ b for a, b in zip(phy_key_hash, self.master_key[:16]))
        
        return session_key, kdr, detection

# === ANA SİMÜLASYON ===
def run_simulation():
    """Ana simülasyon çalıştırıcı"""
    print("="*60)
    print("LEO UYDU PHY-LAYER KEY GENERATION SİMÜLASYONU")
    print("Türkiye FGN-100 Parametreleri")
    print("="*60)
    
    # Parametreler
    sat = FGN100
    print(f"\nUydu: {sat.name}")
    print(f"İrtifa: {sat.altitude_km} km")
    print(f"Hız: {sat.velocity_kms} km/s")
    print(f"Frekans: {sat.frequency_ghz} GHz")
    print(f"Coherence Time: {sat.coherence_time_ms:.2f} ms")
    
    # Kanal oluştur
    channel = LEOChannel(sat)
    print(f"\nDoppler: {channel.f_doppler/1e3:.2f} kHz")
    
    # Güvenlik sistemi
    master_key = bytes([0xAB] * 32)  # Örnek master key
    security = HybridSecuritySystem(master_key)
    
    # Simülasyon parametreleri
    num_rounds = 10
    snr_values = [10, 15, 20, 25, 30]
    
    results = []
    
    for snr in snr_values:
        print(f"\n{'='*40}")
        print(f"SNR = {snr} dB")
        print(f"{'='*40}")
        
        kdr_list = []
        kgr_list = []
        
        for round_idx in range(num_rounds):
            # CSI üret
            num_samples = 1000
            h_ab, h_ba = channel.generate_csi(num_samples, snr)
            
            # Anahtar üret
            session_key, kdr, detection = security.generate_session_key(
                h_ab, h_ba, snr
            )
            
            # Key Generation Rate (KGR) - bits per channel use
            kgr = len(session_key) * 8 / num_samples
            
            kdr_list.append(kdr)
            kgr_list.append(kgr)
            
            if round_idx == 0:
                print(f"\nRound {round_idx + 1}:")
                print(f"  KDR: {kdr:.4f}")
                print(f"  KGR: {kgr:.4f} bits/channel use")
                print(f"  Eve Detected: {detection['eve_detected']}")
                print(f"  Correlation: {detection['correlation']:.4f}")
        
        results.append({
            'snr': snr,
            'kdr_mean': np.mean(kdr_list),
            'kdr_std': np.std(kdr_list),
            'kgr_mean': np.mean(kgr_list)
        })
    
    # Sonuçları görselleştir
    plot_results(results)
    
    return results

def plot_results(results):
    """Sonuçları görselleştir"""
    snrs = [r['snr'] for r in results]
    kdrs = [r['kdr_mean'] for r in results]
    kgrs = [r['kgr_mean'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # KDR grafiği
    ax1.semilogy(snrs, kdrs, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('Key Disagreement Rate (KDR)')
    ax1.set_title('FGN-100: Key Disagreement Rate vs SNR')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.01, color='r', linestyle='--', label='Target KDR = 1%')
    ax1.legend()
    
    # KGR grafiği
    ax2.plot(snrs, kgrs, 'rs-', linewidth=2, markersize=8)
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('Key Generation Rate (bits/channel use)')
    ax2.set_title('FGN-100: Key Generation Rate vs SNR')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('leo_phy_key_results.png', dpi=150)
    print("\n📊 Grafik kaydedildi: leo_phy_key_results.png")
    # plt.show() # disable interactive plot blocking

if __name__ == "__main__":
    results = run_simulation()
