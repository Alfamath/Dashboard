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
- Satu Distribution Map yang selalu menampilkan alokasi akhir MODI.
- Template CSV koordinat untuk 22 sumber dan 38 negara tujuan.
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
4. Buka Distribution Map.
5. Unduh template koordinat, isi latitude/longitude yang sudah diverifikasi, kemudian upload CSV.
6. Peta tunggal akan menampilkan alokasi akhir MODI secara otomatis.

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

Koordinat pabrik/fasilitas dan negara tujuan harus disediakan pada CSV dengan kolom:
`jenis,node,latitude,longitude`.

Nama node harus sama dengan workbook. Jika sebuah node merupakan agregasi sejumlah fasilitas, koordinat representatif harus diberi keterangan agar tidak disalahartikan sebagai lokasi operasional yang pasti. Garis pada peta merupakan penghubung geografis lurus, bukan lintasan aktual kapal/pesawat. Tile OpenStreetMap wajib menampilkan atribusi OpenStreetMap contributors dan digunakan sesuai kebijakan tile server.
