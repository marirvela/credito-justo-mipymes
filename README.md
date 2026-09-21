# ¿Cuánto debería costar mi crédito?

Modelo de referencia de precios del crédito empresarial en Colombia para micro,
pequeñas y medianas empresas, construido con datos abiertos de la
Superintendencia Financiera.

**App desplegada:** [click aqui url streamlit](https://credito-justo-mipymes-9qvejf3foptekgrbtxzrpr.streamlit.app/)

## El problema

Una empresa colombiana que necesita un crédito no tiene forma de saber cuál es
una tasa justa. En la misma semana, las entidades vigiladas desembolsaron
créditos empresariales con tasas entre el 11 % y el 89 % efectivo anual. Una
parte grande de esa diferencia no depende del crédito en sí: la misma
microempresa puede pagar 65 % en una línea y 17 % en la misma línea cuando el
banco la financia con recursos de redescuento.

Las mipymes son el 99,5 % de las empresas formales del país y generan el 79 %
del empleo. Según EMICRON 2024, el 42,5 % de los micronegocios que no pidieron
crédito no lo hicieron por miedo a endeudarse, y el 22,9 % de los que sí
pidieron acudieron a prestamistas informales.

## Los datos

- **Fuente principal:** Tasas de interés activas por tipo de crédito
  (datos.gov.co, `w9zh-vetq`). Una semana completa por mes, julio 2025 –
  junio 2026: 1.454.868 registros crudos, 620.020 tras la limpieza,
  1,16 millones de créditos, COP 103,9 billones.
- **Fuente complementaria:** Tasa de interés bancario corriente (`pare-7x5i`),
  que fija el tope legal de usura de cada modalidad.

**Variable objetivo:** la razón `tasa / tope de usura`, no la tasa. Cada
modalidad tiene su propio tope (1,5 × TIBC), de modo que la tasa cruda mezcla
en qué modalidad cae el crédito con qué tan cerca del máximo lo cobran.

## Resultados

| Hipótesis | Resultado |
|---|---|
| H1 · Tamaño y antigüedad | **Confirmada.** En crédito ordinario: micro 0,74 → pequeña 0,65 → mediana 0,55 → gran empresa 0,47 del tope |
| H2 · Garantías | **Refutada.** Sin garantía y con FNG cuestan igual. Solo el FAG abarata, y porque viene con redescuento |
| H3 · Redescuento | **Confirmada.** Es la palanca mayor: del 65 % al 17 % anual en el mismo producto |
| H4 · Entidad | **Confirmada.** Segunda variable más importante del modelo; para el mismo producto van de 0,50 a 0,97 del tope |
| H5 · Equidad | **Confirmada: no hay prima.** 0,25 puntos entre hombres y mujeres, con signo contrario al esperado. La brecha cruda de 5,5 puntos era composición |

**Conclusión central:** la inequidad de este mercado no está en el precio sino
en el acceso. Una microempresaria paga más que un empresario mediano, pero no
porque sea mujer: porque el producto al que accede tiene un tope legal tres
veces más alto.

## Modelos

Dos variantes con la misma arquitectura:

- **Modelo de mercado** — incluye sexo, grupo étnico y departamento.
- **Modelo solo-riesgo** — los excluye.

La diferencia entre ambos mide la prima que el mercado cobra más allá de lo que
explican las características del crédito. La app muestra las dos cifras juntas.

`HistGradientBoostingRegressor` (`learning_rate=0.04`, `max_iter=600`,
`max_leaf_nodes=63`, `min_samples_leaf=20`), R² = 0,906, error medio de
1,0 puntos de tasa en crédito ordinario y 3,0–3,3 en microcrédito. Se eligió
sobre Random Forest, que tenía 0,0009 menos de RMSE, por un costo de
entrenamiento ocho veces menor.

## Estructura

- `Proyecto1_Chowdhury_Serna_Rodriguez.ipynb` — informe técnico completo
- `streamlit_app.py` — aplicación web
- `requirements.txt` — dependencias, alineadas con el entorno de entrenamiento
- `modelo_mercado.joblib`, `modelo_riesgo.joblib`, `config_app.joblib` — artefactos

## Cómo reproducir el cuaderno

Los datos están en la carpeta `data/`:

| Archivo | Contenido |
|---|---|
| `credito_empresarial_raw.parquet` | 1.454.868 registros crudos, 12 cortes semanales (jul 2025 – jun 2026) |
| `credito_empresarial_clean.parquet` | 620.020 registros tras las exclusiones por régimen de precio |
| `tibc_2025_2026.csv` | Tasa de interés bancario corriente certificada por modalidad |
| `topes_usura_mensual.csv` | Topes de usura derivados (1,5 × TIBC), mes × modalidad |

### Opción 1 — Google Colab (recomendada)

1. Abrir `Proyecto1_Chowdhury_Rodriguez.ipynb` en Colab.
2. Ejecutar la primera celda (instala `pyarrow` e importa las librerías).
3. Ejecutar la segunda celda. Busca los datos en Google Drive; si no los
   encuentra, abre un selector para subirlos desde el computador. Descargar
   los cuatro archivos de `data/` y seleccionarlos ahí.
4. Ejecutar el resto de arriba a abajo.

Alternativa sin subir archivos a mano — clonar el repositorio dentro de Colab
y apuntar `DATA_DIR` a la carpeta, ejecutando esto **antes** de la celda de
carga de datos:

```python
!git clone https://github.com/USUARIO/credito-justo-mipymes.git
DATA_DIR = "/content/credito-justo-mipymes/data"
```

(reemplazar `USUARIO` por el propietario del repositorio)

### Opción 2 — Local

Clonar el repositorio y ejecutar el cuaderno desde la raíz: la celda de carga
detecta `./data` automáticamente. Requiere Python 3.12 y las dependencias de
`requirements.txt`, más `pyarrow`, `matplotlib`, `seaborn` y `jupyter`.

### Tiempos de ejecución

El ciclo completo de los seis modelos tarda unos **34 minutos** en Colab con
CPU (el Random Forest se lleva 997 segundos de ese total). La búsqueda de
hiperparámetros añade unos 15 minutos y la importancia por permutación otros
pocos. Las celdas del EDA corren en segundos.

El orden de ejecución es de arriba a abajo sin saltos. La sección 4
(preparación) aparece antes de la sección 3 (EDA) de forma deliberada, porque
el análisis exploratorio se hace sobre el conjunto limpio.

## Limitaciones

Solo cubre entidades vigiladas: el crédito informal, donde se financian las
empresas más vulnerables, es invisible. Las tasas están censuradas por el tope
de usura, de modo que el modelo estima el precio observado y no el latente. El
73 % de los registros no informa el grupo étnico.

No es asesoría financiera ni una cotización.

## Autores

Nameer Chowdhury, Ricardo Serna y Mariana Rodríguez
Técnicas de Aprendizaje de Máquina · Pontificia Universidad Javeriana · 2026
