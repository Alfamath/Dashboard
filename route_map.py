"""Peta rute pengiriman SCMS (Leaflet + OpenStreetMap) untuk dashboard Streamlit.

Dipakai dari app.py:

    from route_map import build_map_html, default_coordinates
    html, missing = build_map_html(supply, demand, costs, per_method, actual_cost, coords)
    streamlit.components.v1.html(html, height=780)

per_method : {'NWC': {'initial': DataFrame, 'final': DataFrame, 'iterations': int | None,
                      'reference': bool}, ...}
Peta memuat Leaflet dari unpkg dan tile dari OpenStreetMap, jadi butuh internet.
"""
import json

import numpy as np

EPS = 1e-9

# Lokasi representatif (lat, lon). Sumber = kota kantor pusat / pabrik utama kelompok produsen,
# bukan alamat pabrik persis. Tujuan = ibu kota / kota utama negara. Ubah di sini bila perlu.
KOORDINAT_SUMBER = {
    'SCMS RDC': (-26.20, 28.05, 'Johannesburg'),
    'Aurobindo (India)': (17.39, 78.49, 'Hyderabad'),
    'Mylan/Matrix (India)': (17.50, 78.30, 'Hyderabad'),
    'Alere/Inverness (Japan)': (35.68, 139.69, 'Tokyo'),
    'Hetero (India)': (17.30, 78.60, 'Hyderabad'),
    'Cipla (India)': (19.08, 72.88, 'Mumbai'),
    'Strides (India)': (12.97, 77.59, 'Bengaluru'),
    'AbbVie/Abbott (Eropa)': (50.11, 8.68, 'Frankfurt'),
    'Trinity Biotech (Irlandia)': (53.20, -6.10, 'Bray'),
    'KHB (China)': (31.23, 121.47, 'Shanghai'),
    'Standard Diagnostics (Korea)': (37.57, 126.98, 'Seoul'),
    'GSK': (51.49, -0.31, 'Brentford'),
    'Chembio (USA)': (40.82, -73.00, 'Medford, NY'),
    'Micro Labs (India)': (13.10, 77.45, 'Bengaluru'),
    'Emcure (India)': (18.52, 73.86, 'Pune'),
    'MSD': (40.67, -74.29, 'Kenilworth, NJ'),
    'BMS (Eropa)': (48.86, 2.35, 'Paris'),
    'Ranbaxy (India)': (28.46, 77.03, 'Gurugram'),
    'Novartis': (47.56, 7.59, 'Basel'),
    'Orasure (USA)': (40.63, -75.37, 'Bethlehem, PA'),
    'Roche': (47.50, 7.70, 'Basel'),
    'Janssen': (51.32, 4.87, 'Beerse'),
}

KOORDINAT_TUJUAN = {
    'Nigeria': (9.06, 7.49, 'Abuja'), 'Zambia': (-15.39, 28.32, 'Lusaka'),
    'Mozambique': (-25.97, 32.57, 'Maputo'), 'South Africa': (-25.75, 28.19, 'Pretoria'),
    'Zimbabwe': (-17.83, 31.05, 'Harare'), "Côte d'Ivoire": (5.36, -4.01, 'Abidjan'),
    'Tanzania': (-6.79, 39.21, 'Dar es Salaam'), 'Uganda': (0.35, 32.58, 'Kampala'),
    'Rwanda': (-1.95, 30.06, 'Kigali'), 'Vietnam': (21.03, 105.85, 'Hanoi'),
    'Haiti': (18.54, -72.34, 'Port-au-Prince'), 'Ethiopia': (9.03, 38.74, 'Addis Ababa'),
    'Cameroon': (3.85, 11.50, 'Yaoundé'), 'Kenya': (-1.29, 36.82, 'Nairobi'),
    'Ghana': (5.60, -0.19, 'Accra'), 'Namibia': (-22.56, 17.08, 'Windhoek'),
    'Swaziland': (-26.32, 31.14, 'Mbabane'), 'Congo, DRC': (-4.44, 15.27, 'Kinshasa'),
    'Malawi': (-13.96, 33.77, 'Lilongwe'), 'Guyana': (6.80, -58.16, 'Georgetown'),
    'Burundi': (-3.38, 29.36, 'Bujumbura'), 'Botswana': (-24.65, 25.91, 'Gaborone'),
    'Dominican Republic': (18.49, -69.93, 'Santo Domingo'), 'South Sudan': (4.85, 31.58, 'Juba'),
    'Benin': (6.37, 2.39, 'Cotonou'), 'Pakistan': (33.69, 73.05, 'Islamabad'),
    'Guatemala': (14.63, -90.51, 'Guatemala City'), 'Sudan': (15.50, 32.56, 'Khartoum'),
    'Togo': (6.13, 1.22, 'Lomé'), 'Angola': (-8.84, 13.23, 'Luanda'),
    'Libya': (32.89, 13.19, 'Tripoli'), 'Sierra Leone': (8.48, -13.23, 'Freetown'),
    'Lesotho': (-29.31, 27.48, 'Maseru'), 'Mali': (12.64, -8.00, 'Bamako'),
    'Liberia': (6.30, -10.80, 'Monrovia'), 'Guinea': (9.64, -13.58, 'Conakry'),
    'Afghanistan': (34.53, 69.17, 'Kabul'), 'Senegal': (14.72, -17.47, 'Dakar'),
}



# Basis data negara: nama|alias;alias|ibu kota|lat|lon  (dipakai untuk mencocokkan node otomatis)
_NEGARA = """
Algeria||Algiers|36.75|3.06
Angola||Luanda|-8.84|13.23
Benin||Porto-Novo|6.50|2.63
Botswana||Gaborone|-24.65|25.91
Burkina Faso||Ouagadougou|12.37|-1.52
Burundi||Bujumbura|-3.38|29.36
Cameroon|Kamerun|Yaounde|3.85|11.50
Cape Verde|Cabo Verde|Praia|14.93|-23.51
Central African Republic||Bangui|4.36|18.56
Chad||N'Djamena|12.13|15.06
Comoros||Moroni|-11.70|43.26
Congo|Republic of the Congo;Congo Brazzaville;Congo Republic|Brazzaville|-4.27|15.28
Congo, DRC|DRC;DR Congo;Congo DRC;Democratic Republic of the Congo;Congo Kinshasa;Congo Democratic Republic|Kinshasa|-4.44|15.27
Cote d'Ivoire|Ivory Coast;Pantai Gading|Yamoussoukro|6.83|-5.29
Djibouti||Djibouti|11.59|43.15
Egypt|Mesir|Cairo|30.04|31.24
Equatorial Guinea||Malabo|3.75|8.78
Eritrea||Asmara|15.32|38.93
Eswatini|Swaziland|Mbabane|-26.32|31.14
Ethiopia|Etiopia|Addis Ababa|9.03|38.74
Gabon||Libreville|0.39|9.45
Gambia||Banjul|13.45|-16.58
Ghana||Accra|5.60|-0.19
Guinea||Conakry|9.64|-13.58
Guinea-Bissau||Bissau|11.86|-15.60
Kenya||Nairobi|-1.29|36.82
Lesotho||Maseru|-29.31|27.48
Liberia||Monrovia|6.30|-10.80
Libya|Libia|Tripoli|32.89|13.19
Madagascar||Antananarivo|-18.88|47.51
Malawi||Lilongwe|-13.96|33.77
Mali||Bamako|12.64|-8.00
Mauritania||Nouakchott|18.08|-15.98
Mauritius||Port Louis|-20.16|57.50
Morocco|Maroko|Rabat|34.02|-6.83
Mozambique|Mozambik|Maputo|-25.97|32.57
Namibia||Windhoek|-22.56|17.08
Niger||Niamey|13.51|2.11
Nigeria||Abuja|9.06|7.49
Rwanda||Kigali|-1.95|30.06
Sao Tome and Principe|Sao Tome|Sao Tome|0.34|6.73
Senegal||Dakar|14.72|-17.47
Seychelles||Victoria|-4.62|55.45
Sierra Leone||Freetown|8.48|-13.23
Somalia||Mogadishu|2.05|45.32
South Africa|Afrika Selatan;RSA|Pretoria|-25.75|28.19
South Sudan|Sudan Selatan|Juba|4.85|31.58
Sudan||Khartoum|15.50|32.56
Tanzania|United Republic of Tanzania|Dodoma|-6.16|35.75
Togo||Lome|6.13|1.22
Tunisia|Tunisia|Tunis|36.81|10.18
Uganda||Kampala|0.35|32.58
Zambia||Lusaka|-15.39|28.32
Zimbabwe||Harare|-17.83|31.05
Afghanistan||Kabul|34.53|69.17
Armenia||Yerevan|40.18|44.51
Azerbaijan||Baku|40.41|49.87
Bahrain||Manama|26.23|50.59
Bangladesh||Dhaka|23.81|90.41
Bhutan||Thimphu|27.47|89.64
Brunei||Bandar Seri Begawan|4.89|114.94
Cambodia|Kamboja|Phnom Penh|11.56|104.93
China|Tiongkok;PRC;People's Republic of China|Beijing|39.90|116.41
Georgia||Tbilisi|41.72|44.79
India||New Delhi|28.61|77.21
Indonesia||Jakarta|-6.21|106.85
Iran||Tehran|35.69|51.39
Iraq|Irak|Baghdad|33.31|44.36
Israel||Jerusalem|31.77|35.21
Japan|Jepang|Tokyo|35.68|139.69
Jordan|Yordania|Amman|31.95|35.93
Kazakhstan||Astana|51.17|71.43
Kuwait||Kuwait City|29.38|47.99
Kyrgyzstan||Bishkek|42.87|74.59
Laos||Vientiane|17.97|102.63
Lebanon|Libanon|Beirut|33.89|35.50
Malaysia||Kuala Lumpur|3.14|101.69
Maldives||Male|4.18|73.51
Mongolia||Ulaanbaatar|47.89|106.91
Myanmar|Burma|Naypyidaw|19.76|96.08
Nepal||Kathmandu|27.72|85.32
North Korea|Korea Utara|Pyongyang|39.04|125.76
Oman||Muscat|23.59|58.41
Pakistan||Islamabad|33.69|73.05
Philippines|Filipina|Manila|14.60|120.98
Qatar||Doha|25.29|51.53
Saudi Arabia|Arab Saudi|Riyadh|24.71|46.68
Singapore|Singapura|Singapore|1.35|103.82
South Korea|Korea Selatan;Korea;Republic of Korea|Seoul|37.57|126.98
Sri Lanka||Colombo|6.93|79.86
Syria|Suriah|Damascus|33.51|36.29
Tajikistan||Dushanbe|38.56|68.79
Thailand||Bangkok|13.76|100.50
Timor-Leste|East Timor;Timor Leste|Dili|-8.56|125.57
Turkey|Turkiye;Turki|Ankara|39.93|32.86
Turkmenistan||Ashgabat|37.96|58.33
United Arab Emirates|UAE;Uni Emirat Arab|Abu Dhabi|24.45|54.38
Uzbekistan||Tashkent|41.30|69.24
Vietnam|Viet Nam|Hanoi|21.03|105.85
Yemen||Sanaa|15.37|44.19
Albania||Tirana|41.33|19.82
Austria||Vienna|48.21|16.37
Belarus||Minsk|53.90|27.57
Belgium|Belgia|Brussels|50.85|4.35
Bosnia and Herzegovina|Bosnia|Sarajevo|43.86|18.41
Bulgaria||Sofia|42.70|23.32
Croatia|Kroasia|Zagreb|45.81|15.98
Cyprus||Nicosia|35.19|33.38
Czech Republic|Czechia;Ceko|Prague|50.08|14.44
Denmark|Denmark|Copenhagen|55.68|12.57
Estonia||Tallinn|59.44|24.75
Finland||Helsinki|60.17|24.94
France|Prancis|Paris|48.86|2.35
Germany|Jerman|Berlin|52.52|13.40
Greece|Yunani|Athens|37.98|23.73
Hungary|Hungaria|Budapest|47.50|19.04
Iceland|Islandia|Reykjavik|64.15|-21.94
Ireland|Irlandia|Dublin|53.35|-6.26
Italy|Italia|Rome|41.90|12.50
Kosovo||Pristina|42.66|21.17
Latvia||Riga|56.95|24.11
Lithuania||Vilnius|54.69|25.28
Luxembourg||Luxembourg|49.61|6.13
Malta||Valletta|35.90|14.51
Moldova||Chisinau|47.01|28.86
Montenegro||Podgorica|42.44|19.26
Netherlands|Belanda;Holland|Amsterdam|52.37|4.90
North Macedonia|Macedonia|Skopje|42.00|21.43
Norway|Norwegia|Oslo|59.91|10.75
Poland|Polandia|Warsaw|52.23|21.01
Portugal||Lisbon|38.72|-9.14
Romania|Rumania|Bucharest|44.43|26.10
Russia|Russian Federation;Rusia|Moscow|55.76|37.62
Serbia||Belgrade|44.79|20.45
Slovakia||Bratislava|48.15|17.11
Slovenia||Ljubljana|46.06|14.51
Spain|Spanyol|Madrid|40.42|-3.70
Sweden|Swedia|Stockholm|59.33|18.07
Switzerland|Swiss|Bern|46.95|7.45
Ukraine|Ukraina|Kyiv|50.45|30.52
United Kingdom|UK;Great Britain;England;Inggris;Britain|London|51.51|-0.13
Argentina||Buenos Aires|-34.60|-58.38
Bahamas||Nassau|25.05|-77.36
Barbados||Bridgetown|13.10|-59.61
Belize||Belmopan|17.25|-88.77
Bolivia||La Paz|-16.50|-68.15
Brazil|Brasil|Brasilia|-15.79|-47.88
Canada|Kanada|Ottawa|45.42|-75.70
Chile||Santiago|-33.45|-70.67
Colombia|Kolombia|Bogota|4.71|-74.07
Costa Rica||San Jose|9.93|-84.08
Cuba|Kuba|Havana|23.11|-82.37
Dominican Republic|Republik Dominika|Santo Domingo|18.49|-69.93
Ecuador||Quito|-0.18|-78.47
El Salvador||San Salvador|13.69|-89.22
Guatemala||Guatemala City|14.63|-90.51
Guyana||Georgetown|6.80|-58.16
Haiti||Port-au-Prince|18.54|-72.34
Honduras||Tegucigalpa|14.07|-87.19
Jamaica||Kingston|17.97|-76.79
Mexico|Meksiko|Mexico City|19.43|-99.13
Nicaragua||Managua|12.11|-86.24
Panama||Panama City|8.98|-79.52
Paraguay||Asuncion|-25.26|-57.58
Peru||Lima|-12.05|-77.04
Suriname||Paramaribo|5.85|-55.20
Trinidad and Tobago|Trinidad|Port of Spain|10.65|-61.51
United States|USA;US;United States of America;Amerika Serikat;America|Washington, D.C.|38.91|-77.04
Uruguay||Montevideo|-34.90|-56.19
Venezuela||Caracas|10.48|-66.90
Australia||Canberra|-35.28|149.13
Fiji||Suva|-18.14|178.44
New Zealand|Selandia Baru|Wellington|-41.29|174.78
Papua New Guinea||Port Moresby|-9.44|147.18
"""


def _norm(t):
    import re
    import unicodedata
    t = unicodedata.normalize('NFKD', str(t)).encode('ascii', 'ignore').decode().lower()
    t = re.sub(r"[^a-z0-9' ]+", ' ', t.replace("'", ''))
    return re.sub(r'\s+', ' ', t).strip()


def _indeks_negara():
    idx = {}
    for baris in _NEGARA.strip().splitlines():
        nama, alias, kota, lat, lon = baris.split('|')
        for a in [nama] + [x for x in alias.split(';') if x]:
            idx[_norm(a)] = (nama, float(lat), float(lon), kota)
    return idx


_IDX_NEGARA = _indeks_negara()


def _cari_negara_tepat(nama):
    return _IDX_NEGARA.get(_norm(nama))


def _cari_negara_dalam_teks(teks):
    """Cari nama negara sebagai kata utuh di dalam teks (mis. 'Sun Pharma (India)')."""
    import re
    t = ' ' + _norm(teks) + ' '
    terbaik = None
    for alias, v in _IDX_NEGARA.items():
        if len(alias) >= 4 or alias in ('uk', 'us', 'usa', 'uae', 'drc', 'prc'):
            if ' ' + alias + ' ' in t and (terbaik is None or len(alias) > len(terbaik[0])):
                terbaik = (alias, v)
    return terbaik[1] if terbaik else None


def resolve_coordinates(nama_sumber, nama_tujuan, base=None):
    """Cocokkan semua node ke koordinat.

    Urutan: (1) koordinat yang sudah ada/bawaan/CSV, (2) nama mirip dengan node bawaan,
    (3) tujuan = nama negara, sumber = negara yang disebut dalam namanya.
    Mengembalikan (coords, status, belum) -> status: daftar dict untuk tabel pengecekan,
    belum: daftar (jenis, node) yang masih perlu diisi manual.
    """
    import difflib
    coords = dict(base if base is not None else default_coordinates())
    bawaan = default_coordinates()
    status, belum = [], []
    for jenis, names in (('sumber', nama_sumber), ('tujuan', nama_tujuan)):
        kunci_bawaan = {_norm(n): n for (j, n) in bawaan if j == jenis}
        for n in names:
            if (jenis, n) in coords:
                v = coords[(jenis, n)]
                status.append({'jenis': jenis, 'node': n, 'sumber koordinat': 'bawaan / CSV',
                               'lokasi': v[2] if len(v) > 2 else '', 'latitude': v[0], 'longitude': v[1]})
                continue
            hasil, cara = None, ''
            mirip = difflib.get_close_matches(_norm(n), list(kunci_bawaan), n=1, cutoff=0.88)
            if mirip:
                v = bawaan[(jenis, kunci_bawaan[mirip[0]])]
                hasil, cara = v, 'nama mirip: ' + kunci_bawaan[mirip[0]]
            if hasil is None:
                neg = _cari_negara_tepat(n) if jenis == 'tujuan' else None
                if neg is None:
                    neg = _cari_negara_dalam_teks(n)
                if neg is None and jenis == 'tujuan':
                    m = difflib.get_close_matches(_norm(n), list(_IDX_NEGARA), n=1, cutoff=0.86)
                    neg = _IDX_NEGARA[m[0]] if m else None
                if neg is not None:
                    hasil, cara = (neg[1], neg[2], 'ibu kota ' + neg[0]), 'negara: ' + neg[0]
            if hasil is None:
                belum.append((jenis, n))
                status.append({'jenis': jenis, 'node': n, 'sumber koordinat': 'BELUM ADA',
                               'lokasi': '', 'latitude': None, 'longitude': None})
            else:
                coords[(jenis, n)] = hasil
                status.append({'jenis': jenis, 'node': n, 'sumber koordinat': cara,
                               'lokasi': hasil[2], 'latitude': hasil[0], 'longitude': hasil[1]})
    return coords, status, belum


def default_coordinates():
    """Koordinat bawaan: {('sumber'|'tujuan', nama): (lat, lon, kota)}."""
    out = {('sumber', n): v for n, v in KOORDINAT_SUMBER.items()}
    out.update({('tujuan', n): v for n, v in KOORDINAT_TUJUAN.items()})
    return out


def _aliran(X, sumber, tujuan, C, blocked):
    out = []
    for i, j in zip(*np.where(X > EPS)):
        out.append({'s': sumber[i], 't': tujuan[j], 'kg': float(X[i, j]),
                    'c': float(C[i, j]), 'blk': bool(blocked[i, j])})
    out.sort(key=lambda f: -f['kg'])
    return out


def build_data(supply, demand, costs, per_method, actual_cost=None, coords=None):
    """Susun data JSON untuk peta. Mengembalikan (data, daftar_node_tanpa_koordinat)."""
    coords = coords or default_coordinates()
    sumber, tujuan = list(costs.index), list(costs.columns)
    hilang = [n for n in sumber if ('sumber', n) not in coords] + \
             [n for n in tujuan if ('tujuan', n) not in coords]
    blocked = costs.isna().to_numpy()
    C = costs.fillna(0.0).to_numpy(dtype=float)
    res = {}
    for k, r in per_method.items():
        X0 = r['initial'].reindex(index=sumber, columns=tujuan).to_numpy(dtype=float)
        X1 = r['final'].reindex(index=sumber, columns=tujuan).to_numpy(dtype=float)
        res[k] = {
            'awal': _aliran(X0, sumber, tujuan, C, blocked),
            'modi': _aliran(X1, sumber, tujuan, C, blocked),
            'iter': None if r.get('iterations') is None else int(r['iterations']),
            'ref': bool(r.get('reference', False)),
            'cost_awal': float((X0 * C).sum()),
            'cost_awal_valid': float((X0[~blocked] * C[~blocked]).sum()),
            'kg_blk': float(X0[blocked].sum()),
            'cost_modi': float((X1 * C).sum()),
        }
    maks = max([f['kg'] for r in res.values() for t in ('awal', 'modi') for f in r[t]] or [1.0])

    def titik(names, jenis, series):
        out = []
        for n in names:
            if (jenis, n) in coords:
                v = coords[(jenis, n)]
                out.append({'n': n, 'kg': float(series[n]), 'lat': float(v[0]), 'lon': float(v[1]),
                            'kota': v[2] if len(v) > 2 else ''})
        return out

    data = {
        'meta': {'aktual': None if not actual_cost else float(actual_cost), 'maxKg': float(maks),
                 'totalKg': float(np.sum(supply.to_numpy(dtype=float))),
                 'nSrc': len(sumber), 'nDst': len(tujuan)},
        'src': titik(sumber, 'sumber', supply),
        'dst': titik(tujuan, 'tujuan', demand),
        'res': res,
    }
    return data, hilang


def build_map_html(supply, demand, costs, per_method, actual_cost=None, coords=None):
    data, hilang = build_data(supply, demand, costs, per_method, actual_cost, coords)
    # rute yang ujungnya tidak punya koordinat dibuang agar JavaScript tidak error
    ada_s = {s['n'] for s in data['src']}
    ada_t = {d['n'] for d in data['dst']}
    for r in data['res'].values():
        for t in ('awal', 'modi'):
            r[t] = [f for f in r[t] if f['s'] in ada_s and f['t'] in ada_t]
    return TEMPLATE.replace('__DATA__', json.dumps(data, ensure_ascii=False)), hilang


TEMPLATE = r'''<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Peta Rute Pengiriman SCMS</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
:root{
  --panel:#ffffff; --ink:#1d2733; --muted:#5d6b7a; --line:#d9dfe6; --accent:#1f5fa8; --accent-ink:#ffffff;
  --warn:#c62828; --warn-bg:#fdecea; --ok:#1b7f4b; --chip:#eef2f6; --shadow:0 2px 14px rgba(20,35,55,.22);
}
@media (prefers-color-scheme:dark){:root{
  --panel:#18202a; --ink:#e6ebf1; --muted:#9aa8b8; --line:#2c3846; --accent:#6aa9ee; --accent-ink:#0d1722;
  --warn:#ff8a80; --warn-bg:#3a1f1f; --ok:#6fd49e; --chip:#243040; --shadow:0 2px 14px rgba(0,0,0,.5);}}
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:var(--ink);background:var(--panel)}
#map{position:absolute;inset:0}
#panel{position:absolute;z-index:1000;top:12px;left:12px;width:340px;max-width:calc(100% - 24px);max-height:calc(100% - 24px);
  display:flex;flex-direction:column;background:var(--panel);border-radius:10px;box-shadow:var(--shadow);overflow:hidden}
#panel header{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:12px 14px 8px}
#panel h1{font-size:15px;margin:0;line-height:1.25}
#panel h1 small{display:block;font-weight:400;color:var(--muted);font-size:12px;margin-top:2px}
#toggle{border:1px solid var(--line);background:var(--chip);color:var(--ink);border-radius:6px;padding:4px 9px;cursor:pointer;font:inherit;font-size:12px}
#body{overflow-y:auto;padding:0 14px 14px;min-height:0}
#panel.closed #body{display:none}
.lbl{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:12px 0 5px}
.seg{display:flex;border:1px solid var(--line);border-radius:7px;overflow:hidden}
.seg button{flex:1;border:0;background:transparent;color:var(--ink);padding:7px 4px;cursor:pointer;font:inherit;font-weight:600}
.seg button+button{border-left:1px solid var(--line)}
.seg button[aria-pressed=true]{background:var(--accent);color:var(--accent-ink)}
button:focus-visible,.row:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.stats{display:grid;grid-template-columns:1fr auto;gap:3px 12px;font-variant-numeric:tabular-nums}
.stats dt{color:var(--muted)} .stats dd{margin:0;text-align:right;font-weight:600}
.warn{background:var(--warn-bg);color:var(--warn);border-radius:7px;padding:7px 9px;margin-top:8px;font-size:12.5px}
.ok{color:var(--ok)}
.bar{height:9px;border-radius:5px;background:linear-gradient(90deg,#7ad151,#22a884,#2a788e,#414487,#440154)}
.scale{display:flex;justify-content:space-between;font-size:11.5px;color:var(--muted);margin-top:3px;font-variant-numeric:tabular-nums}
.keys{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:12px;color:var(--muted);margin-top:8px}
.keys i{display:inline-block;vertical-align:middle;margin-right:5px}
.k-line{width:22px;border-top:3px solid var(--accent)} .k-blk{width:22px;border-top:3px dashed #d32f2f}
.k-opt{width:22px;border-top:5px solid #0072B2} .k-non{width:22px;border-top:5px solid #F5A300}
.k-src{width:10px;height:10px;background:#e8731a;border:1px solid #fff} .k-dst{width:10px;height:10px;border-radius:50%;background:#2a6f97;border:1px solid #fff}
.opts{display:flex;flex-wrap:wrap;gap:8px 16px;font-size:13px}
.opts label{display:flex;align-items:center;gap:6px;cursor:pointer}
#focusbar{display:none;align-items:center;justify-content:space-between;gap:8px;background:var(--chip);border-radius:7px;padding:6px 9px;margin-top:10px;font-size:12.5px}
#focusbar button{border:0;background:transparent;color:var(--accent);cursor:pointer;font:inherit;font-weight:600}
.list{border:1px solid var(--line);border-radius:7px;max-height:210px;overflow-y:auto}
.row{display:grid;grid-template-columns:1fr auto;gap:0 10px;padding:6px 9px;cursor:pointer;border-bottom:1px solid var(--line);font-size:12.5px;font-variant-numeric:tabular-nums}
.row:last-child{border-bottom:0} .row:hover{background:var(--chip)}
.row b{font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.row span{color:var(--muted);text-align:right}
.row.blk b{color:var(--warn)}
.row small{grid-column:1/-1;color:var(--muted);font-size:11.5px}
.note{font-size:11.5px;color:var(--muted);margin-top:10px}
.leaflet-tooltip{font:12.5px/1.4 system-ui,sans-serif}
.leaflet-tooltip b{font-weight:600}
@keyframes alir{to{stroke-dashoffset:-24}}
.anim{stroke-dasharray:8 6;animation:alir 1.4s linear infinite}
@media (prefers-reduced-motion:reduce){.anim{animation:none}}
@media (max-width:560px){#panel{top:auto;bottom:0;left:0;right:0;width:auto;max-width:none;border-radius:12px 12px 0 0;max-height:62%}}
</style>
</head>
<body>
<div id="map" role="application" aria-label="Peta rute pengiriman SCMS"></div>
<aside id="panel">
  <header>
    <h1>Rute pengiriman SCMS<small id="sub"></small></h1>
    <button id="toggle" aria-expanded="true">Sembunyikan</button>
  </header>
  <div id="body">
    <div class="lbl">Metode solusi awal</div>
    <div class="seg" id="seg-metode" role="group" aria-label="Metode">
        </div>
    <div class="lbl">Tahap</div>
    <div class="seg" id="seg-tahap" role="group" aria-label="Tahap">
      <button data-t="awal" aria-pressed="false">Solusi awal</button>
      <button data-t="modi" aria-pressed="true">Setelah MODI</button>
    </div>
    <div class="lbl">Warna garis</div>
    <div class="seg" id="seg-warna" role="group" aria-label="Warna garis">
      <button data-w="opt" aria-pressed="true">Optimal / tidak</button>
      <button data-w="biaya" aria-pressed="false">Biaya per kg</button>
    </div>
    <div class="lbl">Ringkasan</div>
    <dl class="stats" id="stats"></dl>
    <div id="warn"></div>
    <div id="focusbar"><span id="focustxt"></span><button id="focusclear">Tampilkan semua</button></div>
    <div id="costscale" style="display:none">
      <div class="lbl">Biaya per kg (USD)</div>
      <div class="bar"></div>
      <div class="scale"><span id="cmin"></span><span id="cmax"></span></div>
    </div>
    <div class="keys" id="keys"></div>
    <div class="lbl">Tampilan</div>
    <div class="opts">
      <label><input type="checkbox" id="o-anim" checked> Animasi aliran</label>
      <label><input type="checkbox" id="o-label"> Label negara</label>
    </div>
    <div class="lbl">Rute terbesar</div>
    <div class="list" id="list"></div>
    <p class="note">Klik penanda sumber atau tujuan untuk menyorot rutenya. Lokasi sumber adalah kota representatif kelompok produsen, bukan alamat pabrik. Garis mengikuti lintasan lingkaran besar, bukan jalur kapal atau pesawat sebenarnya.</p>
  </div>
</aside>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const D = __DATA__;
const fmt = (x, d=0) => x.toLocaleString('id-ID', {minimumFractionDigits:d, maximumFractionDigits:d});
const usd = x => 'USD ' + fmt(x);
document.getElementById('sub').textContent = D.meta.nSrc + ' sumber, ' + D.meta.nDst + ' tujuan, ' + fmt(D.meta.totalKg/1e6, 2) + ' juta kg';
const STOPS = ['#7ad151','#22a884','#2a788e','#414487','#440154'];
const BLK = '#d32f2f';

const map = L.map('map', {worldCopyJump:true, minZoom:2, zoomSnap:.25}).setView([8, 38], 3);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom:18, attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'}).addTo(map);

const srcBy = Object.fromEntries(D.src.map(s => [s.n, s]));
const dstBy = Object.fromEntries(D.dst.map(s => [s.n, s]));

// skala warna biaya per kg (log), dari semua rute valid
const valid = [];
for (const r of Object.values(D.res)) for (const t of ['awal','modi']) for (const f of r[t]) if (!f.blk) valid.push(f.c);
const cLo = Math.min(...valid), cHi = Math.max(...valid);
document.getElementById('cmin').textContent = fmt(cLo, 2);
document.getElementById('cmax').textContent = fmt(cHi, 2);
function hex(h){return [1,3,5].map(i => parseInt(h.slice(i,i+2),16));}
function warna(c){
  const u = Math.min(1, Math.max(0, (Math.log(c)-Math.log(cLo))/(Math.log(cHi)-Math.log(cLo))));
  const p = u*(STOPS.length-1), i = Math.min(STOPS.length-2, Math.floor(p)), f = p-i;
  const a = hex(STOPS[i]), b = hex(STOPS[i+1]);
  return 'rgb(' + a.map((v,k) => Math.round(v+(b[k]-v)*f)).join(',') + ')';
}
const lebar = kg => 1.6 + 9*Math.sqrt(kg/D.meta.maxKg);

// status optimal: sebuah rute "optimal" bila selnya dipakai pada solusi akhir MODI metode yang sama
const OPT = '#0072B2', NON = '#F5A300';
let modeWarna = 'opt';
const kunci = (s,t) => s + '\u0001' + t;
const optKg = {};
for (const m of Object.keys(D.res)) optKg[m] = new Map(D.res[m].modi.map(f => [kunci(f.s,f.t), f.kg]));
const kgOpt = f => optKg[metode].get(kunci(f.s,f.t)) || 0;
const diOpt = f => kgOpt(f) > 0;
function warnaGaris(f){
  if (f.blk) return BLK;
  if (modeWarna === 'biaya') return warna(f.c);
  return diOpt(f) ? OPT : NON;
}
function legenda(){
  const k = document.getElementById('keys');
  const biaya = modeWarna === 'biaya';
  document.getElementById('costscale').style.display = biaya ? 'block' : 'none';
  k.innerHTML = (biaya
    ? '<span><i class="k-line"></i>Rute valid, tebal = volume</span>'
    : '<span><i class="k-opt"></i>Optimal (sel dipakai di solusi akhir)</span>' +
      (tahap === 'awal' ? '<span><i class="k-non"></i>Tidak optimal (dipindahkan MODI)</span>' : '')) +
    '<span><i class="k-blk"></i>Rute terblokir (Big-M)</span>' +
    (biaya ? '' : '<span>Tebal = volume</span>') +
    '<span><i class="k-src"></i>Sumber</span><span><i class="k-dst"></i>Tujuan</span>';
}

// lintasan lingkaran besar; dipecah di garis tanggal internasional
function busur(p, q, n=72){
  const r = Math.PI/180, f1=p[0]*r, l1=p[1]*r, f2=q[0]*r, l2=q[1]*r;
  const d = 2*Math.asin(Math.sqrt(Math.sin((f2-f1)/2)**2 + Math.cos(f1)*Math.cos(f2)*Math.sin((l2-l1)/2)**2));
  if (d < 1e-3) return [[p, q]];
  const segs = []; let cur = [], prev = null;
  for (let i=0; i<=n; i++){
    const t = i/n, A = Math.sin((1-t)*d)/Math.sin(d), B = Math.sin(t*d)/Math.sin(d);
    const x = A*Math.cos(f1)*Math.cos(l1) + B*Math.cos(f2)*Math.cos(l2);
    const y = A*Math.cos(f1)*Math.sin(l1) + B*Math.cos(f2)*Math.sin(l2);
    const z = A*Math.sin(f1) + B*Math.sin(f2);
    const pt = [Math.atan2(z, Math.hypot(x,y))/r, Math.atan2(y,x)/r];
    if (prev !== null && Math.abs(pt[1]-prev) > 180){ segs.push(cur); cur = []; }
    cur.push(pt); prev = pt[1];
  }
  segs.push(cur);
  return segs;
}

const layerRute = L.layerGroup().addTo(map);
const layerTitik = L.layerGroup().addTo(map);
const layerLabel = L.layerGroup();
let metode = 'VAM', tahap = 'modi', fokus = null, garis = [];

// penanda
const maxSup = Math.max(...D.src.map(s=>s.kg)), maxDem = Math.max(...D.dst.map(d=>d.kg));
for (const s of D.src){
  const z = Math.round(10 + 10*Math.sqrt(s.kg/maxSup));
  const m = L.marker([s.lat, s.lon], {icon:L.divIcon({className:'', iconSize:[z,z], iconAnchor:[z/2,z/2],
    html:`<div style="width:${z}px;height:${z}px;background:#e8731a;border:2px solid #fff;box-shadow:0 0 0 1px rgba(0,0,0,.45)"></div>`}), keyboard:false});
  m.bindTooltip(`<b>${s.n}</b><br>${s.kota}<br>Supply ${fmt(s.kg)} kg`);
  m.on('click', () => setFokus({tipe:'s', n:s.n}));
  m.addTo(layerTitik);
}
for (const d of D.dst){
  const m = L.circleMarker([d.lat, d.lon], {radius:4+9*Math.sqrt(d.kg/maxDem), color:'#fff', weight:1.5, fillColor:'#2a6f97', fillOpacity:.95});
  m.bindTooltip(`<b>${d.n}</b><br>${d.kota}<br>Demand ${fmt(d.kg)} kg`);
  m.on('click', () => setFokus({tipe:'t', n:d.n}));
  m.addTo(layerTitik);
  L.marker([d.lat, d.lon], {interactive:false, icon:L.divIcon({className:'', iconSize:[0,0],
    html:`<div style="transform:translate(8px,-8px);font:600 11px system-ui;color:#12313f;text-shadow:0 0 3px #fff,0 0 3px #fff,0 0 3px #fff;white-space:nowrap">${d.n}</div>`})}).addTo(layerLabel);
}

function aktif(f){ return !fokus || (fokus.tipe==='s' ? f.s===fokus.n : f.t===fokus.n); }

function gambar(){
  layerRute.clearLayers(); garis = [];
  const r = D.res[metode], flows = r[tahap];
  const anim = document.getElementById('o-anim').checked;
  // rute terkecil dulu agar yang besar berada di atas
  [...flows].sort((a,b) => (modeWarna==='opt' ? (diOpt(a)-diOpt(b)) : 0) || a.kg-b.kg).forEach(f => {
    const s = srcBy[f.s], d = dstBy[f.t], on = aktif(f);
    const pl = L.polyline(busur([s.lat,s.lon],[d.lat,d.lon]), {
      color: warnaGaris(f), weight: lebar(f.kg), opacity: on ? (f.blk ? .95 : .82) : .1,
      dashArray: f.blk ? '9 7' : null, lineCap:'round', className: (anim && on && !f.blk) ? 'anim' : ''});
    pl.bindTooltip(`<b>${f.s}</b> → <b>${f.t}</b><br>${fmt(f.kg)} kg<br>USD ${fmt(f.c,2)}/kg${f.blk ? ' <b style="color:#d32f2f">(rute terblokir, biaya Big-M)</b>' : ''}<br>Biaya ${usd(f.kg*f.c)}` +
      (diOpt(f) ? `<br><b style="color:${OPT}">Optimal</b>${tahap==='awal' ? ' (solusi akhir: ' + fmt(kgOpt(f)) + ' kg)' : ''}` : `<br><b style="color:#b26a00">Tidak optimal</b> (tidak dipakai di solusi akhir)`), {sticky:true});
    pl.on('mouseover', () => pl.setStyle({weight: lebar(f.kg)+2.5, opacity:1}));
    pl.on('mouseout', () => pl.setStyle({weight: lebar(f.kg), opacity: on ? (f.blk ? .95 : .82) : .1}));
    pl.addTo(layerRute); garis.push({f, pl});
  });
  legenda(); ringkas(r); daftar(flows);
}

function ringkas(r){
  const st = document.getElementById('stats'), w = document.getElementById('warn');
  const awal = tahap === 'awal';
  const rows = awal ? [
    ['Biaya rute valid', usd(r.cost_awal_valid)],
    ['Biaya + Big-M', usd(r.cost_awal)],
    ['Sel teralokasi', fmt(D.res[metode].awal.length)],
    ['Sel yang tetap optimal', fmt(r.awal.filter(diOpt).length) + ' dari ' + fmt(r.awal.length)],
    ['Selisih vs optimal', (r.kg_blk>0 ? 'tidak layak' : '+' + fmt((r.cost_awal_valid/r.cost_modi-1)*100, 2) + '%')],
  ] : [
    ['Biaya optimal', usd(r.cost_modi)],
    ['Iterasi MODI', r.iter === null ? '—' : fmt(r.iter)],
    ['Sel teralokasi', fmt(r.modi.length)],
    ['Hemat vs aktual', D.meta.aktual ? fmt((1-r.cost_modi/D.meta.aktual)*100, 2) + '%' : '—'],
  ];
  st.innerHTML = rows.map(x => `<dt>${x[0]}</dt><dd>${x[1]}</dd>`).join('');
  if (awal && r.kg_blk > 0){
    const n = r.awal.filter(f=>f.blk).length;
    w.innerHTML = `<div class="warn">${n} rute terblokir terisi, total ${fmt(r.kg_blk)} kg (${fmt(r.kg_blk/D.meta.totalKg*100,1)}% volume). Solusi awal ini belum layak; MODI memindahkan seluruhnya.</div>`;
  } else if (awal){
    w.innerHTML = `<div class="warn ok" style="background:transparent;padding:7px 0 0">Solusi awal sudah layak: tidak ada rute terblokir.</div>`;
  } else {
    w.innerHTML = `<div class="warn ok" style="background:transparent;padding:7px 0 0">Optimal: tidak ada rute terblokir, semua biaya reduksi ≥ 0.</div>`;
    if (r.ref) w.innerHTML += `<div class="note">MODI dari solusi awal ini tidak dapat dijalankan (basis awal bukan pohon perentang), jadi yang ditampilkan adalah solusi optimal dari MODI yang dimulai dari VAM.</div>`;
  }
}

function daftar(flows){
  const el = document.getElementById('list');
  const rows = flows.filter(aktif).slice(0, 40);
  el.innerHTML = rows.map((f,i) => `<div class="row${f.blk?' blk':''}" tabindex="0" data-i="${flows.indexOf(f)}">
    <b>${f.s} → ${f.t}</b><span>${fmt(f.kg)} kg</span><small>USD ${fmt(f.c,2)}/kg${f.blk?' (terblokir)':''}</small></div>`).join('')
    || '<div class="row"><b>Tidak ada rute</b></div>';
  el.querySelectorAll('.row[data-i]').forEach(row => {
    const go = () => {
      const f = D.res[metode][tahap][+row.dataset.i];
      const g = garis.find(x => x.f === f); if (!g) return;
      map.fitBounds(g.pl.getBounds(), {padding:[60,60], maxZoom:6}); g.pl.openTooltip(g.pl.getBounds().getCenter());
    };
    row.addEventListener('click', go);
    row.addEventListener('keydown', e => { if (e.key==='Enter' || e.key===' '){ e.preventDefault(); go(); } });
  });
}

function setFokus(x){
  fokus = (fokus && x && fokus.tipe===x.tipe && fokus.n===x.n) ? null : x;
  const bar = document.getElementById('focusbar');
  bar.style.display = fokus ? 'flex' : 'none';
  if (fokus) document.getElementById('focustxt').textContent = (fokus.tipe==='s' ? 'Sumber: ' : 'Tujuan: ') + fokus.n;
  gambar();
}

function pasang(id, attr, set){
  document.querySelectorAll('#'+id+' button').forEach(b => b.addEventListener('click', () => {
    set(b.dataset[attr]);
    document.querySelectorAll('#'+id+' button').forEach(x => x.setAttribute('aria-pressed', x===b));
    gambar();
  }));
}
if (!D.res[metode]) metode = Object.keys(D.res)[0];
document.getElementById('seg-metode').innerHTML = Object.keys(D.res).map(k => `<button data-m="${k}" aria-pressed="${k===metode}">${k}</button>`).join('');
pasang('seg-metode', 'm', v => metode = v);
pasang('seg-tahap', 't', v => tahap = v);
pasang('seg-warna', 'w', v => modeWarna = v);
document.getElementById('focusclear').addEventListener('click', () => setFokus(null));
document.getElementById('o-anim').addEventListener('change', gambar);
document.getElementById('o-label').addEventListener('change', e => e.target.checked ? layerLabel.addTo(map) : map.removeLayer(layerLabel));
document.getElementById('toggle').addEventListener('click', e => {
  const p = document.getElementById('panel'), closed = p.classList.toggle('closed');
  e.target.textContent = closed ? 'Tampilkan' : 'Sembunyikan'; e.target.setAttribute('aria-expanded', !closed);
});
if (innerWidth <= 560) document.getElementById('toggle').click();
gambar();
</script>
</body>
</html>
'''