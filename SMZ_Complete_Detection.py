"""
SMZ Defekt Aniqlash Tizimi - To'liq Versiya
Kameradan real-time defekt aniqlash va vizualizatsiya
"""

import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import pandas as pd
import cv2
import time
from datetime import datetime

# Vizual sozlamalar
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 10

class DefectDetectionSystem:
    def __init__(self, model_path='yolov8n.pt', camera_source=0):
        """
        Defekt aniqlash tizimini ishga tushirish
        
        Args:
            model_path: YOLO model fayl yo'li
            camera_source: Kamera manbayi (0=USB, yoki RTSP URL)
        """
        self.model = YOLO(model_path)
        self.camera_source = camera_source
        self.results_list = []
        self.frame_count = 0
        self.defect_colors = {
            0: (255, 0, 0),    # Qizil - Defekt turi 1
            1: (0, 255, 0),    # Yashil - Defekt turi 2
            2: (0, 0, 255),    # Ko'k - Defekt turi 3
        }
        
    def process_video(self, max_frames=None, save_video=False):
        """
        Video oqimini qayta ishlash va defektlarni aniqlash
        
        Args:
            max_frames: Maksimal kadrlar soni (None = cheksiz)
            save_video: Video saqlansinmi
        """
        cap = cv2.VideoCapture(self.camera_source)
        
        if not cap.isOpened():
            print("❌ XATO: Kamera ochilmadi!")
            return
        
        print("✅ Kamera muvaffaqiyatli ochildi")
        
        # Video parametrlari
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video saqlash
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('output_detection.mp4', fourcc, fps, (width, height))
        
        print(f"📹 Video: {width}x{height} @ {fps}fps")
        print("🎯 Aniqlash boshlandi... (Chiqish uchun 'q' tugmasini bosing)")
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Kadr olinmadi!")
                break
            
            # YOLO deteksiyasi
            results = self.model(frame, verbose=False)
            annotated_frame = results[0].plot()
            
            # Statistika qo'shish
            defect_count = len(results[0].boxes)
            
            # Har bir obyektni qayta ishlash
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Natijalarni saqlash
                self.results_list.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'frame': self.frame_count,
                    'class': cls,
                    'confidence': conf,
                    'x1': x1, 'y1': y1, 'x2': y2, 'y2': y2,
                    'area': (x2-x1) * (y2-y1)
                })
                
                # Defekt rangini tanlash
                color = self.defect_colors.get(cls, (255, 255, 255))
                
                # Chizish
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                label = f"Defekt {cls}: {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Statistika paneli
            elapsed_time = time.time() - start_time
            info_text = [
                f"Kadr: {self.frame_count}",
                f"Defektlar: {defect_count}",
                f"FPS: {self.frame_count/elapsed_time:.1f}",
                f"Vaqt: {elapsed_time:.1f}s"
            ]
            
            y_offset = 30
            for text in info_text:
                cv2.putText(annotated_frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                y_offset += 30
            
            # Ko'rsatish
            cv2.imshow("SMZ Defekt Aniqlash Tizimi", annotated_frame)
            
            # Video saqlash
            if save_video:
                out.write(annotated_frame)
            
            # Chiqish
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n🛑 Foydalanuvchi tomonidan to'xtatildi")
                break
            
            self.frame_count += 1
            
            # Maksimal kadrlar
            if max_frames and self.frame_count >= max_frames:
                print(f"\n✅ Maksimal kadrlar soniga yetildi: {max_frames}")
                break
        
        # Tozalash
        cap.release()
        if save_video:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 Jami {self.frame_count} kadr qayta ishlandi")
        print(f"📊 Jami {len(self.results_list)} defekt topildi")
        
    def save_results(self, filename='detection_results.csv'):
        """Natijalarni CSV faylga saqlash"""
        if not self.results_list:
            print("⚠️ Saqlanadigan natijalar yo'q")
            return
        
        df = pd.DataFrame(self.results_list)
        df.to_csv(filename, index=False)
        print(f"💾 Natijalar saqlandi: {filename}")
        return df
    
    def visualize_statistics(self, df=None):
        """
        Defekt statistikasini vizualizatsiya qilish
        """
        if df is None:
            if not self.results_list:
                print("⚠️ Vizualizatsiya uchun ma'lumot yo'q")
                return
            df = pd.DataFrame(self.results_list)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('SMZ Defekt Aniqlash Statistikasi', fontsize=16, fontweight='bold')
        
        # 1. Defektlar soni (class bo'yicha)
        class_counts = df['class'].value_counts()
        axes[0, 0].bar(class_counts.index, class_counts.values, color='steelblue')
        axes[0, 0].set_xlabel('Defekt turi')
        axes[0, 0].set_ylabel('Soni')
        axes[0, 0].set_title('Defekt turlari bo\'yicha taqsimot')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Ishonch darajasi taqsimoti
        axes[0, 1].hist(df['confidence'], bins=30, color='coral', edgecolor='black')
        axes[0, 1].set_xlabel('Ishonch darajasi')
        axes[0, 1].set_ylabel('Chastota')
        axes[0, 1].set_title('Ishonch darajasi taqsimoti')
        axes[0, 1].axvline(df['confidence'].mean(), color='red', 
                           linestyle='--', label=f'O\'rtacha: {df["confidence"].mean():.2f}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Defekt maydoni
        axes[0, 2].hist(df['area'], bins=30, color='lightgreen', edgecolor='black')
        axes[0, 2].set_xlabel('Maydon (piksel²)')
        axes[0, 2].set_ylabel('Chastota')
        axes[0, 2].set_title('Defekt maydoni taqsimoti')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Vaqt bo'yicha defektlar
        frame_counts = df.groupby('frame').size()
        axes[1, 0].plot(frame_counts.index, frame_counts.values, color='purple', linewidth=2)
        axes[1, 0].set_xlabel('Kadr raqami')
        axes[1, 0].set_ylabel('Defektlar soni')
        axes[1, 0].set_title('Vaqt bo\'yicha defektlar')
        axes[1, 0].fill_between(frame_counts.index, frame_counts.values, alpha=0.3)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Heatmap - defekt joylashuvi
        heatmap_data = df.groupby(['class', 'frame']).size().unstack(fill_value=0)
        if not heatmap_data.empty:
            sns.heatmap(heatmap_data, cmap='YlOrRd', ax=axes[1, 1], cbar_kws={'label': 'Soni'})
            axes[1, 1].set_xlabel('Kadr')
            axes[1, 1].set_ylabel('Defekt turi')
            axes[1, 1].set_title('Defekt turi va vaqt heatmap')
        
        # 6. Statistik ma'lumot
        axes[1, 2].axis('off')
        stats_text = f"""
        📊 UMUMIY STATISTIKA
        
        Jami kadrlar: {df['frame'].max() + 1}
        Jami defektlar: {len(df)}
        
        O'rtacha ishonch: {df['confidence'].mean():.3f}
        Minimal ishonch: {df['confidence'].min():.3f}
        Maksimal ishonch: {df['confidence'].max():.3f}
        
        O'rtacha maydon: {df['area'].mean():.0f} px²
        Minimal maydon: {df['area'].min():.0f} px²
        Maksimal maydon: {df['area'].max():.0f} px²
        
        Defekt turlari soni: {df['class'].nunique()}
        """
        axes[1, 2].text(0.1, 0.5, stats_text, fontsize=11, 
                       verticalalignment='center', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('defect_statistics.png', dpi=300, bbox_inches='tight')
        print("📊 Statistika grafigi saqlandi: defect_statistics.png")
        plt.show()
    
    def generate_report(self, df=None):
        """Hisobot yaratish"""
        if df is None:
            if not self.results_list:
                print("⚠️ Hisobot uchun ma'lumot yo'q")
                return
            df = pd.DataFrame(self.results_list)
        
        print("\n" + "="*60)
        print("📋 DEFEKT ANIQLASH HISOBOTI")
        print("="*60)
        print(f"📅 Sana: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎥 Kamera manbayi: {self.camera_source}")
        print(f"🤖 Model: {self.model.model_name}")
        print("-"*60)
        print(f"📊 Jami kadrlar: {df['frame'].max() + 1}")
        print(f"🔍 Jami defektlar: {len(df)}")
        print(f"📈 Kadr boshiga defektlar: {len(df)/(df['frame'].max() + 1):.2f}")
        print("-"*60)
        print("🏷️ DEFEKT TURLARI:")
        for cls, count in df['class'].value_counts().items():
            percentage = (count / len(df)) * 100
            print(f"   Turi {cls}: {count} ({percentage:.1f}%)")
        print("-"*60)
        print(f"✅ O'rtacha ishonch darajasi: {df['confidence'].mean():.3f}")
        print(f"📐 O'rtacha defekt maydoni: {df['area'].mean():.0f} px²")
        print("="*60 + "\n")


def main():
    """Asosiy dastur"""
    print("🚀 SMZ Defekt Aniqlash Tizimi ishga tushirilmoqda...\n")
    
    # Tizimni yaratish
    system = DefectDetectionSystem(
        model_path='yolov8n.pt',  # yoki 'yolov8s.pt' aniqroq natija uchun
        camera_source=0  # 0=USB kamera, yoki RTSP URL
    )
    
    # Video qayta ishlash
    system.process_video(
        max_frames=None,  # Cheksiz (q tugmasi bilan to'xtatish)
        save_video=False   # True - video saqlash
    )
    
    # Natijalarni saqlash
    df = system.save_results('detection_results.csv')
    
    # Statistikani ko'rsatish
    if df is not None and not df.empty:
        system.visualize_statistics(df)
        system.generate_report(df)
    
    print("✅ Dastur muvaffaqiyatli yakunlandi!")


if __name__ == "__main__":
    main()
