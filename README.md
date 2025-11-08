
# Pitch Perfect

En enkel og visuell pitch-detektor og audio-analyseapplikasjon. Programmet leser lyd fra mikrofonen, analyserer frekvensinnholdet i sanntid, identifiserer dominante frekvenser/noter og tilbyr flere plot-funksjoner (spektrogram, FFT, tidsdomene).

Funksjoner
- Sanntids audioopptak (PyAudio)
- FFT-basert frekvensdeteksjon med kvarttoner/cent-beregning
- GUI (PyQt5) med knapper for å starte/stoppe og plotte siste rammer eller flere sekunder
- Ferdig bygget Windows .exe i `./Application/Pitch Perfect.exe`, denne er det bare å installere å kjør.

Krav
- Python 3.11 — 3.14 anbefales
- Windows (prosjektet inkluderer en ferdig .exe for Windows)
- Mikrofon / audio input
- Avhengigheter: se `ApplicationCode/requirements.txt` (bl.a. numpy, scipy, matplotlib, PyQt5, PyAudio)

Installasjon (PowerShell)
1. Opprett og aktiver et virtuelt miljø:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Oppdater pip og installer avhengigheter:

```powershell
python -m pip install --upgrade pip
pip install -r .\ApplicationCode\requirements.txt
```

Merk om PyAudio-feil
På Windows kan `PyAudio` være vanskelig å bygge fra kilde. Hvis `pip install PyAudio` feiler, prøv:

```powershell
pip install pipwin
pipwin install pyaudio
```

Eller last ned en passende PyAudio-wheel for din Python-versjon fra en pålitelig kilde og installer med `pip install <wheel-file>`.

Kjøring
- Kjør fra kildekode (utvikling):

```powershell
python .\ApplicationCode\PitchPerfectTest.py
```

- Kjør ferdigbygget applikasjon (.exe):

```powershell
start .\Application\"Pitch Perfect.exe"
```

Bruk
- Start/Stop: bruk knappene "Start Audio Processing" / "Stop Audio Processing" for å aktivere/deaktivere sanntidsanalyse.
- Plot-knapper: genererer matplotlib-plot av siste FFT-ramme, spektrogram eller tidsdomene for de siste N sekundene.
- Notefremheving: GUI viser navnet på den mest dominerende tonen (f.eks. A4) og avvik i cents.

Prosjektstruktur
- `Application/` – ferdigbygget Windows .exe
- `ApplicationCode/` – kildekode (f.eks. `PitchPerfectTest.py`) og `requirements.txt`
- `AudioFiles/` – (valgfritt) eksempelfiler eller testopptak

Vanlige problemer / feilsøking
- Ingen lyd / ingen mikrofon: Sjekk at mikrofonen er aktivert i Windows -> Personvern -> Mikrofon og at programmet har tilgang.
- Lav latens / droppede rammer: prøv å redusere bufferstørrelse eller settrate. PyAudio/driver kan begrense.
- Matplotlib-vinduer åpner seg ikke: sørg for at GUI-tråden ikke blokkeres; bruk plot-knappene i applikasjonen.

Bidra
- Hvis du vil gjøre endringer, lag en Branch og send en pull request. Beskriv endringen og hvordan den kan testes.

Kontakt
- Spørsmål eller forslag kan legges i repository issues.

```


