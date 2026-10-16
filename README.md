# Efecto Imán

Aplicación local para estudiar aperturas fuera de Bandas de Bollinger con velas
de 15 minutos. Analiza simétricamente gaps alcistas (objetivo: banda superior)
y bajistas (objetivo: banda inferior). Descarga datos públicos sin clave mediante
`yfinance`, guarda diagnósticos y velas en SQLite y exporta CSV. No ejecuta
órdenes ni usa contratos de opciones. El scanner en vivo no usa el modelo
experimental de ciencia de datos incluido en este repositorio.

## Inicio

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Abre `http://127.0.0.1:8501`. La aplicación funciona localmente; la consulta
de datos bursátiles requiere Internet. Para cargar hasta 50 sesiones recientes:

```powershell
.\.venv\Scripts\python.exe backfill_history.py
```

## Compartir una copia en Streamlit Community Cloud

El repositorio de GitHub debe contener el código, `app.py`, `requirements.txt`
y `.streamlit/config.toml`. El archivo `.gitignore` excluye la base SQLite,
capturas, registros, copias de respaldo, entorno virtual y secretos locales.
En Community Cloud selecciona este repositorio, la rama publicada y `app.py`
como archivo principal. Selecciona Python 3.11 para coincidir con el entorno
local probado.

La instalación en la nube tiene su propia carpeta `data/`: no ve la base ni
las etiquetas guardadas en esta computadora. La app crea una base nueva y puede
descargar historial desde la interfaz. **Community Cloud no garantiza conservar
archivos locales entre reinicios o redespliegues**, por lo que las revisiones
manuales y los diagnósticos escritos allí se deben exportar si se quieren
conservar. No uses esa copia como capturador fiable de los primeros 30 o 60
segundos: las apps gratuitas pueden hibernar cuando no reciben visitas y
reiniciarse; el corte en vivo exige un proceso activo antes de la apertura.
La instalación local sigue siendo la fuente de las capturas de apertura.

## Revisión manual de gráficas

La tercera sección, **Revisión manual**, muestra hasta 50 sesiones guardadas.
Por defecto enseña aperturas fuera de la banda objetivo con gap de al menos
1 %, aunque el score haya sido bajo. Se puede ampliar a todas las aperturas
fuera de banda o a todos los símbolos. Cada página muestra 12, 24 o 50
gráficas. La vista **Sesión completa** permite juzgar el desarrollo posterior;
las vistas **10 s**, **20 s**, **30 s** y **60 s** muestran sólo precios observados hasta
ese corte y bandas BB(20,2) de 15 minutos recalculadas provisionalmente.
Si falta el corte real, la pantalla indica que no está disponible.

Cada caso admite **Excelente**, **Buena**, **Mala**, **Dudosa** o **Pendiente**. Las etiquetas se
guardan en la tabla `manual_reviews` de SQLite; Pendiente significa que aún no
se ha juzgado, no que sea Mala. El CSV de revisión exporta etiquetas junto con
variables disponibles en cada corte. El score final aparece sólo como columna
de auditoría: no debe usarse para entrenar una predicción temprana. Esta vista
no entrena un modelo ni cambia las alertas.

Los archivos históricos actuales contienen el primer y último negocio de los
primeros 10 s, pero no su trayectoria íntegra ni cortes de 20/30 s. Para
completar esos cortes con operaciones históricas SIP de Alpaca, configura las
dos variables de entorno descritas más abajo y ejecuta:

```powershell
.\.venv\Scripts\python.exe backfill_30s_alpaca.py --days 50
```

El script guarda `apertura_historica_30s_FECHA.csv` y una trayectoria con el
último precio negociado de cada segundo en
`apertura_historica_trayectoria_30s_FECHA.csv`, sin reemplazar los cortes
originales. La gráfica temprana dibuja también Bollinger provisional para
esos segundos cuando existe la trayectoria. Los tiempos de las operaciones
son históricos de mercado; no demuestran que el proveedor las hubiera
entregado al equipo en esos segundos.

## Estudio histórico del score temprano

Ejecuta `.\.venv\Scripts\python.exe evaluate_early_score.py` para repetir la
comparación de datos conocidos en la apertura y a las 09:31 contra la etiqueta
`score_15m >= 70`, calculada al cerrar la primera vela de 15 minutos. El script
lee las sesiones ya guardadas en SQLite, excluye el día en curso y las aperturas
dentro de Bollinger, entrena una regresión logística con fechas anteriores y
evalúa en fechas posteriores. La imputación y la escala se ajustan sólo con
entrenamiento. Guarda probabilidades por símbolo y métricas en `data/snapshots`.

La columna histórica `open` de la primera vela sirve como aproximación al
instante de apertura, pero no es la cotización que el equipo habría recibido
a las 09:30:10. Yahoo vía
`yfinance` ofrece velas históricas desde 1 minuto; ese primer minuto sólo está
completo a las 09:31. El estudio tampoco valida rentabilidad ni toque de banda:
la etiqueta reproduce exclusivamente el score geométrico posterior. La muestra
de 1 minuto es reducida y el universo guardado cambió de 22 a 72 símbolos al
final del periodo. No se alteran los pesos del scanner ni las alertas en vivo a
partir de esta evaluación exploratoria.

La sección Historial muestra la cobertura de los cortes reales de 09:30:10
para las últimas 50 sesiones guardadas y la incluye en el ZIP como
`cobertura_apertura_10s.csv`. Un corte sólo cuenta si existe su archivo de
cotizaciones y la hora de recepción está entre 09:30:00 y 09:30:10 ET.
Esta tabla registra ausencias; no crea precios subminuto. Una prueba directa
del proveedor rechazó `interval=10s`. Sus velas de `1m` sólo pudieron
consultarse dentro de los últimos 30 días, en solicitudes de hasta 8 días.

Para reconstruir los primeros diez segundos de hasta 50 sesiones anteriores,
`backfill_10s_alpaca.py` consulta operaciones históricas SIP de Alpaca. Requiere
una cuenta y dos claves de API locales en las variables de entorno
`APCA_API_KEY_ID` y `APCA_API_SECRET_KEY`; no guarda las claves en archivos ni
ejecuta órdenes. Con ellas configuradas en PowerShell:

```powershell
.\.venv\Scripts\python.exe backfill_10s_alpaca.py --days 50
```

Para revisar una sesión concreta una vez pasados 15 minutos desde el corte,
usa `--date`, y después compárala con el score posterior:

```powershell
.\.venv\Scripts\python.exe backfill_10s_alpaca.py --date 2026-10-12
.\.venv\Scripts\python.exe review_10s_day.py 2026-10-12
```

El segundo comando entrena sólo con fechas anteriores y exporta una tabla de
scores tempranos y finales en `data/snapshots/review_10s_FECHA.csv`.

Cada archivo `data/snapshots/apertura_historica_10s_FECHA.csv` conserva el
primer y último precio negociado entre 09:30:00 y 09:30:10 ET, las horas de
mercado y el número de operaciones por símbolo. Las descargas terminadas se
omiten al repetir el comando; `--refresh` las vuelve a consultar. SPX se omite
porque es un índice sin operaciones propias. El Historial muestra por separado
las operaciones históricas y las cotizaciones **recibidas en vivo**. Una marca
histórica de operación no demuestra que ese dato hubiera llegado al equipo
antes de las 09:30:10, ni que fuera ejecutable a ese precio.

Tras descargar esos archivos, ejecuta este estudio de los diez segundos:

```powershell
.\.venv\Scripts\python.exe evaluate_historical_10s.py
```

Recalcula gap, dirección, distancia y banda provisional de 15 minutos con el
último precio negociado antes de las 09:30:10 y los cierres previos almacenados.
Mide la pendiente previa y el ángulo provisional normalizado a 5 minutos sin
usar velas futuras. Reserva las últimas
15 sesiones para prueba y guarda métricas y predicciones en `data/snapshots`.
No modifica el score ni las alertas de la aplicación. Sólo compara los símbolos
con diagnóstico previo en SQLite. El reporte distingue operaciones con
condición `I` (odd lot), que pueden no actualizar el precio de la vela. La
etiqueta representa el score posterior, no una ganancia.

Para estudiar específicamente los casos con score menor de 70 a los 10 s que
terminan con `score_15m >= 70`, ejecuta:

```powershell
.\.venv\Scripts\python.exe evaluate_rescue_10s.py
```

Compara modelos supervisados con validación cronológica dentro del entrenamiento
y evalúa el elegido en las últimas 15 sesiones. El reporte incluye las alertas
acertadas y omitidas; es un experimento separado del scanner y no cambia sus
pesos automáticamente.

`audit_score_outcomes.py` compara, sin cambiar la operativa, el score con el
reingreso observado a Bollinger bajo varios plazos y movimientos exploratorios.
Exporta una tabla por símbolo y un resumen en `data/snapshots`. Los toques
registrados en la primera vela de 15 minutos no tienen minuto exacto; el
reporte los separa de los que ocurrieron con certeza después de 09:45.

## Corte automático de la operativa diaria

La pantalla muestra relojes de Nueva York y Hermosillo que avanzan cada segundo
mientras la aplicación está abierta.
El reloj de Nueva York se resalta durante el primer minuto de apertura. Un
aviso visible en ambos menús indica cuándo inicia la preparación, cuántos
símbolos tienen bandas previas, si el flujo está conectado, cuántas
cotizaciones válidas llegaron, cuándo se analiza cada corte y la hora en que
termina cada diagnóstico preliminar. Si un corte no llega a ejecutarse o llega
con cero cotizaciones, el aviso lo dice.

Para cada día hábil, el proceso prepara las velas previas de 15 minutos y
conecta el flujo de cotizaciones a las
**09:25 de Nueva York**. Conserva el corte de **09:30:10 ET** para auditoría.
La primera lista visible se calcula a las **09:30:30 ET** con los precios que
el proveedor entregó localmente hasta ese instante. A las **09:31:00 ET**
calcula otra lista independiente con el minuto completo de cotizaciones.
Ambas usan Bollinger BB(20,2) de 15 minutos provisional, con el último precio
conocido en su corte. Mantén Streamlit
ejecutándose y la página abierta antes de las 09:25; si el proceso se inicia
después del corte, no reconstruye artificialmente esas cotizaciones.
El equipo debe permanecer encendido y despierto, con conexión a Internet;
una pestaña abierta sin el servidor Streamlit funcionando no ejecuta el
scanner. Cada lista aparece cuando termina su cálculo y guardado, no
necesariamente en el segundo exacto del corte.
Si Windows está configurado para suspenderse por inactividad, se puede mantener
despierto temporalmente hasta después del corte con
`python keep_awake_until.py 2026-10-13T06:40:00-07:00` (ajusta la fecha).
El ayudante requiere corriente alterna, deja que la pantalla se apague y al
terminar restaura la suspensión normal. Cerrar la tapa o apagar el equipo
puede suspenderlo igualmente.
Cuando la aplicación está abierta antes del día de mercado, el servidor
programa también el siguiente día entre semana en un hilo propio; el corte no
depende de que el navegador actualice justo a las 09:25. Este programador
excluye fines de semana, pero no consulta automáticamente los festivos de la
bolsa.
El reloj comprueba el estado del programador cada segundo para recuperar una
espera suspendida. Si el corte no se capturó, la pantalla lo indica y continúa
actualizando las velas disponibles en segundo plano.

Cada lista usa la **última cotización recibida antes de su propio plazo**,
no necesariamente el precio oficial de apertura. La marca de
tiempo del proveedor puede adelantarse hasta tres segundos a la
recepción local por desfase entre relojes; la recepción local determina si
el mensaje entró al corte. El gap, la
distancia a Bollinger y el score inicial se basan en ese precio, en las
bandas previas cerradas y en la banda provisional del corte. El ángulo dentro
del corte se puntúa si existen dos precios con horas de mercado distintas.
A las 09:31:00 la banda de 15 minutos aún es provisional; no se usa la vela
que cerrará a las 09:45. Las velas completas se actualizan
después de las 09:31. Si una cotización no llega a tiempo, el símbolo queda
sin apertura en el diagnóstico. El corte auditado se guarda en
`data/snapshots/apertura_10s_FECHA.csv`; la sesión se guarda en SQLite cuando
se recibe al menos una cotización. La vista histórica conserva su escaneo
actual y no intenta reconstruir un corte de diez segundos con velas de 1 minuto.

Los archivos `candidatos_apertura_30s_FECHA.csv` y
`candidatos_apertura_60s_FECHA.csv` guardan las dos listas en `data/snapshots`;
los archivos `diagnostico_apertura_30s_FECHA.csv` y
`diagnostico_apertura_60s_FECHA.csv` guardan también los descartados. El flujo
público de Yahoo puede retrasarse u omitir mensajes: no se garantiza precio
para los 72 símbolos en ninguno de los dos cortes.

Para conservar y explicar una captura real, ejecuta
`.\.venv\Scripts\python.exe audit_live_10s.py AAAA-MM-DD` desde esta carpeta.
El comando lee `apertura_10s_AAAA-MM-DD.csv`, recalcula el score de apertura
usando únicamente esas cotizaciones y las velas `PREVIA` guardadas, y compara
los candidatos con el registro escrito en el instante del corte. Guarda la
tabla completa y los candidatos en `data/snapshots` sin sobrescribirlos al
repetirlo. La comparación separada con el diagnóstico posterior sí se actualiza.
En el corte de diez segundos, la distancia exterior y el score usan la banda
BB(20,2) de 15 minutos recalculada provisionalmente con el último precio
recibido antes de 09:30:10 ET. La banda anterior sigue siendo la referencia
para medir la reacción y la compresión previa; su pendiente ya no reemplaza
la pendiente dentro de la apertura. Si llegaron dos cotizaciones con horas de
mercado distintas, el ángulo observado dentro de los diez segundos también
entra al filtro y al score. Si sólo llegó una, ese ángulo queda pendiente y el
score de apertura es preliminar. No equivale a probabilidad de éxito.

Para observar la reacción de Bollinger **después de abrir**, ejecuta
`.\.venv\Scripts\python.exe opening_band_reaction.py AAAA-MM-DD`. Con el precio
recibido a los diez segundos y los 19 cierres previos calcula la banda BB(20,2)
de 15 minutos provisional y su cambio respecto a la banda previa. Ese cambio
es conocido a los diez segundos, pero no es una pendiente entre dos instantes
posteriores a la apertura. A partir de la próxima captura, el flujo Yahoo se
conserva hasta 60 segundos en `apertura_flujo_60s_AAAA-MM-DD.csv`; también se
escribe `apertura_flujo_30s_AAAA-MM-DD.csv` para los análisis existentes. El archivo
`reaccion_bollinger_apertura_AAAA-MM-DD.csv` informa la pendiente de 10, 20 y
30 segundos sólo si llegaron al equipo dos cotizaciones con marcas de mercado
distintas antes de cada plazo; en caso contrario deja la pendiente vacía.

La operativa diaria muestra primero los candidatos del corte de 30 segundos y
los reemplaza por la lista congelada a 1 minuto cuando está lista. Conserva
ambas tablas descargables para comparación. Hasta las 09:35 ET los ordena por
`score_open`; el ángulo posterior de cinco minutos no los elimina de esas
listas. La tabla diagnóstica sigue mostrando
el score de la etapa disponible. Desde las 09:35 la lista vuelve a los
candidatos confirmados por el ángulo inicial. El reloj inicia la actualización
de velas de 1 minuto al llegar a las 09:31; la lista se refresca cuando
termina la descarga y el cálculo, según la latencia real del proveedor. Junto
a cada símbolo se muestra
la primera hora de **registro local** conocida. El archivo
`data/signal_availability.csv` registra hora de inicio y de conclusión de
cada cálculo, fuente y etapa. Empieza a registrar desde que se instala esta
versión; no permite reconstruir horas de aparición anteriores ni demuestra
que el usuario vio la pantalla o que la señal era operable a ese precio.

El estudio de las primeras ventanas de las siete sesiones disponibles está en
`data/snapshots/early_window_report_2026-10-09.md`; se puede repetir con
`analyze_early_window.py` usando las dos descargas CSV guardadas allí.

## Etapas temporales

1. **09:30 ET:** las últimas velas *cerradas* de la sesión previa definen las
   bandas y su pendiente. El precio de apertura permite calcular gap,
   distancia a la banda y z-score. `candidate_at_open` y `score_open` no
   incluyen datos posteriores.
2. **09:31 ET:** al cerrar la primera vela de 1 minuto se guardan OHLC, retorno,
   cuerpo y reversión inicial. `confirmed` vale 1, 0 o -1 (dato no disponible).
   Esa confirmación no es un filtro duro para `candidate`. `score_1m` ya pondera
   el ángulo provisional normalizado de la banda objetivo de ese minuto.
3. **09:35 ET:** se sustituye provisionalmente el cierre de la primera vela de
   15 minutos por el último cierre disponible: velas de 1 minuto para el día
   actual y velas de 5 minutos para fechas históricas. El ángulo 09:30–09:35
   entra en `score_5m` y debe ser ≤ 25° para UP y DOWN. Para DOWN se invierte
   el signo del ángulo de la banda inferior.
4. **09:45 ET:** al cerrar la primera vela de 15 minutos se calculan ancho
   posterior, cambio del ancho, pendiente posterior de la banda objetivo y
   distancia restante. `score_15m` usa el mayor ángulo observado en los tres
   tramos de cinco minutos y penaliza la expansión del ancho. Estos datos solo
   entran al score desde ese momento. `score` muestra la etapa más reciente.

La banda media es la media de 20 cierres de 15 minutos. Las bandas son
`media ± 2 × desviación estándar poblacional`. El ancho previo es
`(superior − inferior) / media × 100`. La pendiente previa de la banda objetivo
es la pendiente de una regresión lineal de sus últimas cinco observaciones,
dividida por el último valor de la banda y expresada en porcentaje por vela.
El cambio previo del ancho usa la misma regresión y se expresa en puntos
porcentuales por vela. El z-score queda vacío si la desviación estándar es cero.

Los límites están centralizados en [scanner_settings.py](scanner_settings.py).
Cada campo de `ScannerThresholds` acepta `None` para desactivarlo. Los
valores por defecto son **puntos de partida para observar distribuciones**, no
parámetros validados. `DIAGNOSTIC` devuelve todos los símbolos con datos de
sesión previa, incluso si fallan condiciones; `FILTERED` devuelve solo
candidatos. `failed_conditions` explica qué límites fallaron y
`pending_conditions` cuáles aún no se pueden evaluar. Una apertura dentro de
la banda objetivo nunca se considera candidata. La columna `candidate` puede
cambiar a las 09:45 si se activa un límite de expansión posterior.
Después de cambiar solo los pesos o la curva del score, ejecuta
`.\.venv\Scripts\python.exe rescore_history.py` para recalcular los scores
guardados sin descargar velas. El script crea un respaldo SQLite. Si cambian
filtros o métricas de mercado, vuelve a ejecutar el escaneo histórico.

El score 0–100 mide solo parecido con la geometría buscada. Sus pesos y escalas
exploratorias están en [scanner_settings.py](scanner_settings.py) y
[opening_scan.py](opening_scan.py). **No es probabilidad de éxito ni predicción
de rentabilidad.** La tabla se ordena por score y conserva todas las métricas,
incluidos los casos sin candidato perfecto.

El ángulo se calcula como `grados(atan(((banda_actual/banda_anterior)-1)
× 100 × 5/minutos_transcurridos))`. `target_angle_score` vale 1 cuando la
banda está plana o cerrándose; disminuye gradualmente hasta 0.4 a 25° y llega
a 0 a 70°. Su peso es 5. La expansión del ancho total pesa 0.75 solo después
del cierre de 09:45; la reversión de la primera vela pesa 0.25.
Los campos `score_*_component` muestran la contribución en puntos de cada
componente al score final.

## Toque de banda y límites de datos

Se busca el primer toque de la banda objetivo después de la señal. Para una
vela de 15 minutos se compara su rango con la banda **conocida al comienzo de
esa vela**, y el evento se conoce al cierre. Si hubo confirmación a las 09:31,
los datos de 1 minuto de 09:31 a 09:44 pueden detectar toques rápidos. Cuando
faltan esos minutos, se omite el rango completo 09:30–09:45 para no atribuir
un toque anterior a la confirmación; `touch_status` deja visible la
incertidumbre. `minutes_until_touch` se mide desde 09:30 o desde 09:31 cuando
hubo confirmación.

Un toque de una banda móvil no prueba que el precio haya revertido. Se guarda
`reversion_to_touch_band_pct` (separación entre apertura y banda al tocarla,
con signo en dirección de reversión) y
`band_at_or_beyond_open_at_touch`. Este último señala que la banda ya alcanzó
o superó el precio de apertura: el toque puede explicarse por desplazamiento
del indicador. La vista muestra por separado esos casos.

El proveedor público puede retrasar, corregir u omitir velas y solo ofrece
histórico de 1 minuto reciente; el ángulo histórico utiliza velas de 5 minutos.
En fechas antiguas `confirmed=-1` cuando falta la primera vela de 1 minuto;
no se interpreta como rechazo. No hay feed de ticks ni
garantía de alerta exactamente a la apertura.

## Archivos

- `app.py`: dashboard, tabla diagnóstica, gráficas e historial.
- `opening_live.py`: programación y captura del corte de diez segundos.
- `opening_scan.py`: métricas temporales, filtros, score y toque de banda.
- `evaluate_early_score.py`: evaluación histórica separada por fechas del score a 09:45.
- `scanner_settings.py`: límites activables y pesos del score.
- `indicators.py`: cálculo de Bollinger.
- `real_data.py`: descarga, zona horaria de Nueva York y velas completas.
- `daily_chart.py`: gráficas de velas, bandas y señales observadas.
- `daily_store.py`: persistencia local en `data/efecto_iman.db`.
- `outcomes.py`: movimientos descriptivos ex post del subyacente.
- `backfill_history.py`: carga de sesiones recientes disponibles.
- `rescore_history.py`: recalcula scores guardados y crea un respaldo local.
- `config.py`: rutas y 72 símbolos configurados (listado original más los nuevos).

Al cambiar el universo, las sesiones guardadas con los 51 símbolos anteriores
permanecen en SQLite, pero la vista Historial muestra solo sesiones recalculadas
para el universo actual. Usa «Obtener histórico reciente de 15 min» para cargar
las nuevas empresas en hasta 50 sesiones disponibles.

La exportación contiene `diagnosticos.csv`, `velas_15m.csv` y
`resultados_descriptivos.csv`. Los resultados descriptivos no son retornos
ejecutables: no modelan comisiones, spreads, préstamo ni contratos.
