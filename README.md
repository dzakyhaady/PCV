# Real Boxing — POV dengan Pygame + OpenCV

Dzaky Haady - 5024241076

Game boxing arcade bergaya pixel dari sudut pandang orang pertama, dengan ritme pertarungan yang terinspirasi game tinju klasik: baca ancang-ancang, hindari atau tangkis, lalu balas saat lawan terbuka. Game dan preview webcam tampil bersama di **satu jendela Pygame**, sedangkan OpenCV memproses gerakan dari kamera. Proyek ini tidak membutuhkan Panda3D.

## Setup di Spyder

Gunakan environment Spyder yang sama. Jika menyiapkan environment lain, instal dependensi berikut dari terminal:

```bash
pip install opencv-python numpy pygame
```

Di Spyder, buka `main.py` lalu Run (F5). Preview kamera muncul di pojok kanan atas jendela game. Tekan `SPACE` untuk mulai dan `ESC` untuk keluar.

Efek suara arcade dibuat sebagai file WAV di `assets/boxing/audio/` dan diputar untuk bel ronde, ancang-ancang lawan, jab, pukulan yang mengenai lawan, pemain terkena pukulan, guard, block, dan dodge. Jika perangkat audio tidak tersedia, game tetap berjalan tanpa suara.

## Kontrol

- Webcam: ayunkan glove ungu kiri/kanan ke arah tengah untuk jab sesuai sisi; rapatkan kedua glove untuk dodge/duck; tahan `B` untuk guard.
- Tanpa webcam: `J`/`F` atau `L`/`H` untuk jab, tahan `B` untuk guard, tahan `A`/`D` untuk dodge.
- Saat dodge, arena bergeser mengikuti arah gerakan sementara lawan tetap tertambat di tengah ring.
- Lawan memberi ancang-ancang sekitar 0,6 detik sebelum pukulan datang, dengan indikator arah serangan kiri/kanan.
- Hindari atau tangkis tepat saat pukulan tiba untuk membuka jendela **COUNTER**. Jab di luar jendela itu akan ditangkis dan tidak mengurangi HP lawan.
- Saat HP pemain mencapai 50%, **Adrenalin** aktif selama 8 detik dan pukulan memberi damage lebih besar. Saat HP lawan mencapai 50%, **Berserk** aktif selama 8 detik sehingga lawan menyerang lebih cepat dan lebih kuat. Keduanya hanya muncul sekali per ronde.
- Tampilan lawan diperbesar dengan zoom POV agar terasa lebih dekat; HUD dan inset kamera tetap berukuran normal.
- Detektor memakai centroid glove ungu. Tiap glove dilacak terpisah di sisi kiri/kanan frame; jab dipicu oleh gerakan cepat ke tengah. Kedua glove yang dirapatkan memicu dodge/duck sekali sampai dipisahkan lagi; `B` tetap tersedia untuk block.

## Catatan

Game memakai aset 8-bit dari `assets/boxing/`. Lawan utama menggunakan `enemy_pov_guard_new.png`, aset terbaru dari pengguna dengan latar transparan. `enemy_punch.png` tampil selama fase pukulan lawan; `enemy_hit_left.png` atau mirror kanan tampil saat counter pemain mengenai lawan, lalu pose siaga kembali setelah efek hit; sarung tangan idle kiri/kanan sekarang terpisah; asset jab hanya menggerakkan tangan yang menyerang sementara tangan lain tetap di pose idle; saat jab, glove penyerang langsung berganti ke sprite pukulan kiri atau mirror kanan tanpa fade, sedangkan tangan lain tetap idle; setelah jab selesai glove kembali idle; latar ring juga berasal dari asset pack. HUD menampilkan bar PLAYER/OPPONENT, ronde, dan timer. Saat dodge, kamera dan arena bergeser sementara lawan tetap di titik ring; inset webcam tidak ikut bergeser. Deteksi gerakan tangan memakai rentang hue ungu pada HSV. Preview kamera berwarna menampilkan overlay hijau dan kotak oranye pada glove ungu yang dilacak. Cahaya putih di foto pengguna tidak masuk mask karena saturasinya rendah.

## Keterkaitan dengan materi PCV

Pipeline citra ada di `MotionTracker.process()` pada `main.py`. Inset webcam ditampilkan langsung pada jendela Pygame agar kamera dan game terlihat bersamaan.

| Materi | Penerapan di game |
|---|---|
| Fundamental | Webcam menghasilkan frame `H x W x 3` bertipe `uint8`; koordinat piksel dari frame dipakai untuk melacak pusat tangan. Resolusi diminta 640 x 480 dan laju 30 FPS. |
| Transformasi intensitas | `cv2.inRange()` memilih piksel jenuh dengan hue ungu pada rentang 112–162 dengan saturasi minimum 48 untuk menghasilkan mask biner. |
| Equalisasi histogram | `cv2.equalizeHist()` meratakan kanal Value hanya untuk preview berwarna; threshold deteksi memakai frame HSV sebelum equalization. |
| Filter spasial | `cv2.GaussianBlur()` dengan kernel 5 x 5 mengurangi noise lokal sebelum thresholding. |
| Model warna | Frame BGR dikonversi ke HSV. Glove dipisahkan berdasarkan warna ungu; kontur kecil/besar disaring, pencarian dibatasi ke pita tangan 8–96% frame, dan tiap sisi memilih kandidat terdekat dengan posisi sebelumnya. |
| Morphology | Opening dengan structuring element kotak 5 x 5 mengurangi bintik kecil; closing mengisi celah kecil pada mask tangan. |

Urutan proses per frame: **akuisisi → Gaussian blur → BGR ke HSV → threshold hue ungu → opening/closing → kontur dan centroid kiri/kanan → klasifikasi jab dan gesture tangan rapat untuk dodge**. Preview kamera menampilkan video berwarna dengan overlay hijau serta kotak oranye pada glove yang terlacak. Kedua glove memicu dodge ketika pusatnya berjarak kurang dari 20% lebar frame dan hampir sejajar.















