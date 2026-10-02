# Lokales OceanScribe Review-Studio

Das Studio setzt die erste Phase des Pro-Plans vom 2. Oktober 2026 um:
begrenzte Textprüfungen und eigene Aufnahmen vor weiteren Trainingsläufen.
Es benötigt keinen Cloud-Key. Der Server bindet ausschließlich an `127.0.0.1`.

## Start und Fortsetzen

Im Projektverzeichnis mit der vorhandenen Trainingsumgebung starten:

```sh
.venv/bin/python -m oceanscribe_cleanup.review_studio
```

Anschließend `http://127.0.0.1:8766` öffnen. Mit `Ctrl+C` stoppen. Derselbe
Startbefehl setzt die Kampagne aus `data/review-studio/current` fort. Statusprüfung:

```sh
.venv/bin/python -m oceanscribe_cleanup.review_studio --check
```

Keinen neuen Datenordner für eine Fortsetzung wählen: Budgets und Entscheidungen
gehören zur gespeicherten Kampagne. Alle Kampagnendaten, Audio und Exporte sind
Git-ignoriert. Die SQLite-Datei und ihre Dateien gemeinsam sichern; während einer
Sicherung den Server stoppen.

## Kurzer erster Durchgang

1. Unter **Prüfen** bewusst einen Fünferblock starten. Zuerst stehen sechs
   priorisierte EN/DE-Reparaturkarten aus drei bekannten Problemfamilien bereit.
   Der Vorschlag ist ein Kandidat, kein menschlich bestätigtes Label.
2. Den unveränderlichen Rohtext mit dem Ziel vergleichen. Bedeutung, Begründungen,
   Zahlen, Negationen und Bezüge erhalten. Korrigieren, übernehmen, zurückstellen
   oder ausschließen. Auch übersprungene Karten zählen als Arbeit.
3. Bei **Development-Referenzen** erscheint zunächst kein Teacher-Ziel. Die eigene
   Referenz zuerst festlegen. Spätere Modellvergleiche zeigen höchstens zwei
   verblindete Antworten und verlangen jeweils eine eigene Bedeutungsbewertung.
4. Unter **Diktieren** oben bei **Aufnahmeaufgabe auswählen** eine Aufgabe und
   Nemotron, Parakeet oder beide auswählen. Unter **Sprechtext** steht der Text,
   der eingesprochen werden soll; bei freien Situationen in eigenen Worten sprechen.
   Mikrofon wählen, lokal aufnehmen, stoppen und anhören. Erst der tatsächliche
   ASR-Rohtext ist der Input; die Sprechvorlage ersetzt ihn nicht.
5. Je ASR-Variante das Cleanup-Ziel bestätigen. Fehlende Fakten, die nur im Audio
   oder Skript stehen, als **ASR-Verlust** ausschließen. Die automatischen
   Abwesenheitsprüfungen für Negationen/Zahlen ersetzen keine Bedeutungsprüfung.

Ein Clip mit zwei ASR-Varianten zählt als ein Aufnahmeversuch. Neu einsprechen
zählt erneut. Retakes und Varianten behalten die Szenariofamilie und den Split.
Browser-Neuladen startet kein Mikrofon. Gespeicherte ASR-Jobs werden fortgesetzt;
eine vor dem Upload abgebrochene Aufnahme bleibt als verbrauchter Versuch sichtbar.
Browser-WebM ohne Dauer im Dateikopf wird anhand der tatsächlich dekodierten
Samples gemessen. Überlange Clips werden abgewiesen, niemals still gekürzt.
**Gespeichertes Audio erneut transkribieren** wiederholt einen technischen Fehler
ohne neue Aufnahme und ohne zusätzlichen Versuch (höchstens drei Retries).
**Diese Aufnahme verwerfen** schließt ungeeignete Aufnahmen aus, ohne Audio zu
löschen oder das Budget zurückzusetzen. Die Auswahl zeigt den Zustand je Aufgabe.
Textentwürfe werden im lokalen Browser gespeichert; endgültige Entscheidungen
liegen versioniert im Studio. Technische Speicherwiederholungen behalten ihre
Request-ID. Konflikte überschreiben keine neuere Revision.

Das Startbudget beträgt **50 Review-Karten und 10 Aufnahmeversuche**. Unter
Fortschritt kann es bewusst auf höchstens **100/20** erweitert werden. Die
Reservierungen sind 20 Reparaturen, 10 Kontrollen, 10 Referenzen und 10 Vergleiche.
Nicht jede Reservierung muss ausgeschöpft werden. Die Aufnahmeplanung enthält pro
Zehnerblock 4 DE, 4 EN und 2 gemischte Aufgaben, verteilt auf 5 Train, 3 Development
und 2 freie, verschlossene Testaufgaben. Die Erweiterungsaufgaben sind andere
Szenarien. EN natürlich mit dem eigenen Akzent sprechen; Mischsprache erhalten.

## Daten, Export und nächste Trainingsphase

Der Initialimport verwendet den eingefrorenen Fidelity-v2-Snapshot mit 3.478
Train-Varianten. Vorher akzeptierte Daten bleiben als **Legacy-Übernahme** mit
`human_verified=false` markiert; das ist kein neuer vollständiger Qualitätsaudit.
Bekannte falsche Familien bleiben bis zur Entscheidung gesperrt. Zehn
stratifizierte Kontrollkarten und zehn geschützte Referenzkarten sind vorbereitet.
Die vervollständigte Luna-CSV wird dabei nicht automatisch importiert.

**Train-Snapshot exportieren** erzeugt einen neuen atomaren Ordner unter
`data/review-studio/current/snapshots/`: bestehende CleanupRecords, kompatibles
Dataset-Manifest und getrennte Provenienz-Sidecars. Nur akzeptierte finale
Train-Revisionen passieren Familien-, Provenienz-, Duplikat- und Längengates.
Dev/Test gelangen weder in Trainings- noch Teacher-Exporte. Eigene Train-Texte
sind standardmäßig für externe Reviews gesperrt; Audio bleibt lokal. Korrekturen
an Elternfällen erneuern geprüfte einfache Ableitungen und sperren andere.
Die lokale Löschfunktion entfernt Audio und abgeleitete Aufnahmeinhalte und
sperrt betroffene lokale Snapshots. Bereits anderweitig kopierte Exporte können
dadurch nicht zurückgerufen werden.

Speichern und Exportieren starten kein Training. Als nächstes müssen Reparaturen
und Referenzen bestätigt, der getrennte Generator/Reviewer-Pilot abgeschlossen
und die Pools eingefroren werden. Erst danach folgt das begrenzte A/B aus dem
v2-Plan: gleicher reparierter Bestand und gleicher Real-ASR-Anteil, nur der neue
Synthetikanteil unterscheidet die Zweige. Kein unbegrenztes Nachtrainieren und
keine Veröffentlichung sind Teil dieser Studio-Implementierung.

Der Prompt bleibt unverändert `raw-v1`, ohne Chat-Template. Das spätere Ziel bleibt
die CPU-Engine `qwen35-cpu` mit `.q35h`; GGUF hier betrifft ausschließlich die
NeMo-ASR-Modelle, nicht den Export des Cleanup-Modells.

### Eingefrorener Vergleich vom 2. Oktober

Die Erfassungsrunde ist abgeschlossen: 31 Review-Aufgaben und zehn
Aufnahmeversuche. Sieben Textreferenzen und drei bestätigte Development-Aufnahmen
ergeben zehn Bewertungsfamilien mit 13 Textpaaren. Zwei ausgelassene und eine
ausgeschlossene Textreferenz werden nicht benutzt; daraus entsteht keine weitere
Pflichtaufgabe. Zehn Modellvergleichskarten sind im Startbudget reserviert.

Der lokale A/B-Snapshot enthält 3.478 reparierte Bestandsvarianten, zehn echte
ASR-Varianten aus fünf Trainingsaufnahmen und 114 geprüfte neue Synthetikfamilien.
Nemotron und Parakeet derselben Aufnahme zählen gemeinsam als eine Familie.
Alle Development-Daten liegen separat unter `protected/snapshots/`; die
Train-Datei enthält ausschließlich Train. Elf neue Kandidaten bleiben
zurückgestellt. Diese Daten sind ein lokaler Versuchsbestand; ein öffentlicher
Korpus und eine allgemeine Trainings-/Veröffentlichungslizenz sind damit nicht
freigegeben.

Beide r16-LoRA-Zweige starten vom ursprünglichen bilingualen Pilotadapter, mit
frischem Optimizer, maximal 100 Schritten und identischer Sprach-/ASR-Verteilung.
Die CLI-Anbindung und Sicherheitsprüfungen sind im README unter
„Controlled Review Studio continuation“ dokumentiert. Für die spätere Auswahl
werden höchstens zwei technische Kandidaten auf den bestätigten Referenzen
verblindet verglichen. Die Modellevaluation darf keine menschliche
Bedeutungsfreigabe aus bloßem WER oder Modellübereinstimmung erzeugen.

## Echte ASR-Integration und Prüfgrenzen

Beide Wege verwenden die vorhandene lokale **NeMo-Speech.cpp**-Bibliothek im
Nachbarprojekt OceanScribe auf CPU. Nemotron nutzt Streaming mit den gefundenen
OceanScribe-Einstellungen; Parakeet TDT nutzt den Offline-Aufruf. Modellrevision,
Quantisierung, Bibliothek, Decoder, Audioformat und echte Hashes stehen in den
ASR-Sidecars. Die Parakeet-Gewichte stammen aus dem offiziellen NVIDIA-Repository
und liegen im lokalen Hugging-Face-Cache. Audio wird lokal mit ffmpeg auf mono
16 kHz dekodiert, nicht über Browser-Spracherkennung.

Ein öffentlicher JFK-Audioclip wurde tatsächlich durch beide Modelle transkribiert.
Dieser Techniktest belegt keine Qualität für die eigene Stimme. Mikrofonaufnahme
und persönliche Transkripte müssen noch am Gerät geprüft werden. Die komplette
Parität der Browser-Audiovorverarbeitung mit dem Produktions-Mikrofonpfad ist
ebenfalls noch nicht belegt; der vorhandene Worker bietet keinen Dateieingang.
ASR-Lizenzen sind kein Freigabenachweis für den gesamten Cleanup-Korpus.

Automatische Bedeutungschecks sind begrenzte Regeln für explizite Annotationen,
keine allgemeine semantische Bewertung. Nicht menschlich bewertete Antworten
bleiben `not_evaluated`; sie erhalten keine erfundene Fehlerquote von null.

## Reproduzierbare Tests

```sh
.venv/bin/python -m pytest -q -m 'not model_download'
.venv/bin/ruff check src/oceanscribe_cleanup/review_studio tests/test_studio_*.py
node --check src/oceanscribe_cleanup/review_studio/static/app.js
```

Optionaler echter ASR-Test mit bereits lokal vorhandenen Modellen:

```sh
OCEANSCRIBE_RUN_ASR_INTEGRATION=1 .venv/bin/python -m pytest tests/test_studio_asr.py -q
```

Technische Browserprüfungen verwenden einen separaten Fixture-Datenordner. Sie
verbrauchen keine Karten oder Aufnahmeversuche der persönlichen Kampagne.
Der lokale Ausführungsbericht liegt unter
`reports/generated/review-studio-implementation-2026-10-02.json`.
