# SCMS Global Transportation Optimization Dashboard

Streamlit dashboard untuk workbook SCMS dengan worksheet:
- `Data_Bersih`
- `Supply_Demand`
- `Matriks_Biaya`

## Fitur
- Upload workbook Excel melalui browser.
- Overview supply, demand, transaksi, dan rute tersedia.
- Tabel transaksi, supply-demand, dan matriks biaya.
- Perhitungan solusi awal Restricted NWC, Restricted LCM, dan Restricted VAM.
- Perbaikan solusi awal menggunakan MODI (u-v) sampai kondisi optimalitas terpenuhi.
- Distribution Map interaktif (Leaflet + OpenStreetMap): pilih metode NWC/LCM/VAM dan tahap solusi awal atau setelah MODI, garis biru untuk rute optimal dan oranye untuk rute solusi awal yang digantikan MODI, serta pilihan pewarnaan biaya per kg.
- Koordinat bawaan (kota representatif) dan opsi unggah CSV koordinat untuk menggantinya.
- Unduh riwayat pivot MODI dan matriks alokasi akhir.

## Menjalankan di Windows PowerShell

Buka terminal di folder proyek:

```powershell
& 'C:\Users\ASUS\AppData\Local\Programs\Python\Python314\python.exe' -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
& '.\.venv\Scripts\python.exe' -m streamlit run app.py
```

Buka alamat lokal yang ditampilkan Streamlit, biasanya `http://localhost:8501`.

## Cara menggunakan
1. Upload workbook Excel di sidebar.
2. Periksa Overview dan Data Management.
3. Buka Algorithm Analysis untuk membandingkan solusi awal NWC, LCM, dan VAM serta melihat proses MODI.
4. Buka Distribution Map; peta langsung tampil dengan koordinat bawaan.
5. (Opsional) Buka panel koordinat, unduh template, ubah latitude/longitude, lalu unggah CSV.

## Metodologi

Sel biaya kosong diperlakukan sebagai rute terlarang, bukan biaya nol. Tiga metode solusi awal menggunakan adaptasi feasibility-aware agar tidak memilih rute kosong dan sisa jaringan tetap mampu memenuhi supply serta demand.

MODI dimulai dari solusi awal VAM. Untuk workbook SCMS yang diperiksa, 22 node sumber dan 38 node tujuan terhubung oleh basis awal sebanyak 59 sel alokasi positif (m+n-1), sehingga membentuk basis pohon untuk perhitungan potensial u dan v. MODI memilih sel nonbasis dengan indeks perbaikan paling negatif, membentuk siklus tertutup, menggeser alokasi sebesar theta, dan mengulang sampai seluruh indeks perbaikan rute yang tersedia tidak negatif.

Hasil pada workbook yang diuji:
- Biaya awal VAM: USD 56,691,715.04
- Biaya akhir MODI: USD 54,461,842.66
- Penurunan biaya: USD 2,229,872.38 (sekitar 3.93%)
- Iterasi MODI: 16
- Volume terpenuhi: 19,550,180 kg
- Rute positif pada solusi akhir: 59

Angka ini adalah hasil matematis dari matriks biaya dalam workbook (USD/kg), bukan biaya aktual yang diverifikasi dari tagihan logistik. Hasil optimalitas berlaku untuk model, biaya, dan rute yang tersedia di workbook tersebut.

## Peta

Peta memakai koordinat bawaan di `route_map.py` (kota representatif, bukan alamat pabrik). Untuk mengganti, unggah CSV dengan kolom:
`jenis,node,latitude,longitude`.

Nama node harus sama dengan workbook. Jika sebuah node merupakan agregasi sejumlah fasilitas, koordinat representatif harus diberi keterangan agar tidak disalahartikan sebagai lokasi operasional yang pasti. Garis pada peta merupakan penghubung geografis lurus, bukan lintasan aktual kapal/pesawat. Tile OpenStreetMap wajib menampilkan atribusi OpenStreetMap contributors dan digunakan sesuai kebijakan tile server.
