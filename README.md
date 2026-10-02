# Proyecto — Clasificación de sentimiento de reseñas (Parte 1: ML clásico)

Aprendizaje de Máquina 2026-20, Universidad de los Andes.

Clasificar reseñas de productos en español en `negativo`, `neutral` o `positivo` usando **solo scikit-learn** y
representaciones clásicas (bag-of-words, TF-IDF, n-gramas). Métrica de Kaggle: **accuracy**.

> Estado al 2 de octubre de 2026: 4 experimentos terminados, 4 envíos generados.
> Mejor modelo según la CV: **experimento 4** (accuracy CV 0,901 · test 0,898).
> Score público de Kaggle conocido: experimento 2 → 0,88 (otros equipos reportan ≈ 0,90).

---

## 1. Resumen de resultados

| # | Notebook | Idea | Clasificador | Acc. CV | Acc. test | Robustez* | Envío | Kaggle público |
|---|---|---|---|---|---|---|---|---|
| 1 | `experimento1.ipynb` | Línea base: TF-IDF (1,2) del texto completo | LR, `C = 10` | 0,736 ± 0,013 | 0,724 | 0,721 | `submission_01.csv` | pendiente |
| 2 | `experimento2.ipynb` | TF-IDF por partes: texto completo + **última oración** + **tras el último conector** | LR, `C = 10` | 0,887 ± 0,010 | **0,899** | 0,686 | `submission_02.csv` | **0,88** |
| 3 | `experimento3.ipynb` | TF-IDF por partes: texto completo + **oraciones con conector conclusivo** (sin usar posición) | LR, `C = 10` | 0,798 ± 0,015 | 0,796 | **0,800** | `submission_03.csv` | pendiente |
| 4 | `experimento4.ipynb` | Como el 2, pero con la **última oración con opinión** (detector de opinión + lista de evasivas) | LinearSVC, `C = 0,2` | **0,901 ± 0,006** | 0,898 | 0,736 | `submission_04.csv` | pendiente |

\* *Robustez:* accuracy en el test con las oraciones de cada reseña desordenadas al azar (mide cuánto depende el
modelo de que el veredicto esté al final). El clasificador y su `C` se eligieron en cada experimento con su propia CV.

**Mejor modelo para la competencia: experimento 4** (gana al experimento 2 en los 5 folds de la CV). En el test la
diferencia con el experimento 2 no se confirma (0,8975 frente a 0,8992, dentro del error estándar ≈ 0,006); el score
público de `submission_04` ayudará a confirmarlo. Cuando suban los envíos, anoten el score en esta tabla y en la
columna `kaggle_publico` del notebook correspondiente.

---

## 2. Estructura del repositorio

```
project_ml_p1/
├── README.md                 # este documento
├── train.csv                 # 12.000 reseñas con etiqueta
├── eval.csv                  # 3.000 reseñas sin etiqueta (solo para predecir; PROHIBIDO entrenar con ellas)
├── sample_submission.csv     # formato de envío
├── experimento1.ipynb        # exploración + preprocesamiento + validación + experimento 1
├── experimento2.ipynb        # experimento 2 (+ prueba de robustez)
├── experimento3.ipynb        # experimento 3 (veredicto por contenido)
├── experimento4.ipynb        # experimento 4 (última oración con opinión + LinearSVC)
├── envios/
│   └── submission_XX.csv     # un archivo por experimento enviado
└── modelos/
    └── modelo_XX.joblib      # pipeline entrenado con las 12.000 reseñas, uno por envío
```

`experimento1.ipynb` es el notebook "base": contiene la **exploración de datos** y la **justificación del
preprocesamiento**. Los demás notebooks solo repiten un resumen de la configuración.

---

## 3. Entorno y cómo ejecutar

| | Versión usada |
|---|---|
| Python | 3.14.6 (Anaconda) |
| scikit-learn | 1.9.0 |
| pandas | 3.0.3 |
| nltk | 3.10.0 (instalado, aún no usado) |

- En VS Code seleccionen el kernel de **Anaconda** (`anaconda3`). El Python del sistema no tiene las librerías.
- Para comprobar resultados: **Restart & Run All**. Ejecutar celdas sueltas o en desorden puede dejar variables
  viejas y dar resultados que no corresponden al notebook.
- Los resultados son **determinísticos**: misma semilla, mismo split y mismos folds. Ejecutar de nuevo un notebook
  completo da exactamente las mismas métricas. Con otras versiones de las librerías podrían cambiar los últimos
  decimales.
- Tiempos aproximados: experimento 1 ≈ 1 min, 2 ≈ 3 min, 3 ≈ 4 min, **4 ≈ 15 min** (el detector se entrena en cada
  fold y la búsqueda compara dos clasificadores).
- **Cargar un `.joblib`:** `joblib` guarda las funciones y clases propias **por referencia**. Antes de cargarlo hay
  que definir las del notebook correspondiente: `dividir_oraciones`, `separar_partes` (modelos 2 y 3) o la clase
  `PartesConOpinion` y sus funciones (`es_evasiva`, `sin_evasivas`, `tras_ultimo_conector`; modelo 4). Además, hay que
  pasarle texto ya normalizado con `normalizar_texto()`.

---

## 4. Plantilla de los notebooks

Cada experimento va en **su propio notebook autocontenido**: `experimentoN.ipynb`. Estructura:

```
# Experimento N — <nombre corto>
  Intro: qué continúa, aviso de que los experimentos anteriores NO se re-ejecutan, contenido.

## 1. Configuración común (resumen)            ← copiar tal cual de experimento2/3/4.ipynb
   1.1 Librerías, semilla y datos               (SEED = 42, CLASES, carga con .copy())
   1.2 Normalización base                       (normalizar_texto, aplicada a train y eval)
   1.3 Validación y métricas                    (split 80/20 estratificado, StratifiedKFold(5), SCORING)
   1.4 Funciones auxiliares                     (tabla_cv, evaluar_en_test, registrar_resultado,
                                                 generar_envio, top_coeficientes)
   + celda con las métricas fijas de experimentos anteriores (REFERENCIA_EN, ACC_FOLDS_EN, ROBUSTEZ_TEST_REF)

## 2. Experimento N — <nombre>
   Motivación (qué problema del experimento anterior ataca) · Hipótesis · Proceso (pasos)
   2.1 Análisis de errores del mejor modelo anterior (cross_val_predict sobre X_train)
   2.2 ... 2.k   Diagnóstico / comparación de variantes con CV sobre X_train (ablación: un cambio a la vez)
                 → después de cada tabla o gráfico: celda markdown "Lectura de ..."
   2.x Modelo del experimento N: GridSearchCV con el clasificador como hiperparámetro,
       cada clasificador con SU PROPIA grilla de C
   2.x Evaluación en test (evaluar_en_test) + coeficientes
   2.x Prueba de robustez (oraciones desordenadas / última oración al inicio)
   2.x Errores que quedan (cross_val_predict sobre X_train, NO sobre el test)
   2.x Registro (registrar_resultado) y envío (generar_envio(busqueda, numero=N))
   2.x Análisis del experimento: resultados, de dónde viene la mejora, alcances y límites, próximos pasos
```

### Reglas que seguimos en todos los experimentos

| Regla | Detalle |
|---|---|
| **Nunca entrenar con `eval.csv`** | Solo se usa para predecir el envío. |
| **Mismo split y semilla** | `SEED = 42`; `train_test_split(test_size=0.20, stratify=y, random_state=SEED)`; `StratifiedKFold(5, shuffle=True, random_state=SEED)`. Así todos los experimentos son comparables. |
| **Roles de los datos** | `X_train` (9.600): búsqueda con CV. `X_test` (2.400): hold-out que **solo se reporta**. Leaderboard privado de Kaggle: test definitivo. |
| **Se decide con CV, no con el test** | Elegir configuraciones e ideas con la accuracy media de CV. Mirar el test para decidir es leakage leve. |
| **Análisis de errores sobre train** | Con `cross_val_predict` sobre `X_train`, nunca con los errores del test. |
| **Todo lo que aprende de los datos va dentro del `Pipeline`** | Vocabulario, IDF y también los modelos auxiliares que usan etiquetas (como el detector de opinión del experimento 4, que se entrena en el `fit` de un transformador). Así cada fold solo ve sus datos de entrenamiento. |
| **Cada experimento ajusta su propio `C`… y su clasificador** | El óptimo cambia con las features. Si se comparan clasificadores, todos con su propia grilla en el mismo `GridSearchCV` (comparar un modelo afinado contra otro con un solo `C` no es justo). |
| **Ablación** | Medir el aporte de cada cambio por separado para saber de dónde viene la mejora. |
| **Métricas** | Se decide con **accuracy** (`refit="accuracy"`); se reporta **F1 macro** como control. |
| **¿Una mejora es real?** | Debe superar la desviación entre folds; si es dudosa, comparar **fold por fold** (mismos folds). Ante un empate, elegir lo más simple. |
| **No re-ejecutar experimentos anteriores** | Sus métricas se escriben como valores fijos. Las configuraciones anteriores solo se incluyen como filas de referencia cuando se mide algo nuevo que no existía en su notebook. |
| **Envío** | `generar_envio()` reentrena con las 12.000 reseñas, valida el formato (`id,answer`, 3.000 filas, 3 etiquetas) y guarda el CSV y el `.joblib`. |
| **Documentación** | Todo en español. Cada decisión, supuesto y descarte se justifica en markdown, incluyendo **por qué no es leakage** y **qué supuesto introduce** una feature. |

### Normalización base (fija en todos los experimentos)

`normalizar_texto()`: reparar mojibake (`aquÃ­` → `aquí`) · minúsculas · quitar tildes **conservando la ñ** ·
letras repetidas 3+ → 2 (`errror` → `error`) · colapsar espacios.

**No** se quitan stopwords (las listas en español incluyen `no`, `pero`, `aunque`, `nada`, que son clave aquí) ni la
puntuación (hace falta para separar oraciones).

---

## 5. Conclusiones hasta ahora

### Exploración (`experimento1.ipynb`, secciones 1–2)
- Clases casi balanceadas: 35,1 % / 29,8 % / 35,1 %. Predecir siempre la mayoritaria da 35,1 %.
- La longitud no distingue clases. Train y eval tienen la misma estructura (≈ 3,9 oraciones por reseña).
- `neutral` son reseñas **descriptivas**: casi sin conectores de juicio y con más cifras y siglas (`USB`, `110 voltios`).
- `negativo` y `positivo` usan los mismos conectores con la misma frecuencia: lo que decide es **qué viene después**.
- Hay ruido inyectado: 507 palabras con y sin tilde, el 40 % del vocabulario aparece una sola vez (muchos errores
  de tipeo como `coomprar`), letras repetidas y mojibake.

### Experimento 1 — Línea base
- `neutral` queda prácticamente resuelta (F1 0,99). **Todo el error está entre `negativo` y `positivo`**: solo
  60,7 % de acierto entre esas dos clases, poco más que el azar.
- Sobreajuste fuerte (train ≈ 1,0), pero la CV es plana entre `C = 3` y `C = 100`: **el límite no es la
  regularización, sino que la bolsa de palabras no ve el orden**.

### Experimento 2 — Final de la reseña
- Diagnóstico: entrenando con una sola parte, la primera oración da 0,51, el texto completo 0,74 y la **última
  oración 0,85**. El veredicto está al final.
- Mejor combinación: texto completo + última oración + tras el último conector, cada parte con su propio TF-IDF
  (gana a "texto + última" en los 5 folds).
- Negativo vs. positivo en test: de 60,7 % a **85,9 %**. Kaggle público: **0,88**.
- **Prueba de robustez:** con las oraciones desordenadas cae a 0,686 y con la última oración al inicio a 0,574,
  **por debajo de la línea base**. Aprendió una regla de **posición** que funciona porque `eval.csv` tiene la misma
  estructura que train, pero no debería llevarse a reseñas de otra fuente sin validarlo.

### Experimento 3 — Veredicto por contenido
- Idea: identificar el veredicto por **cómo empieza la oración** (`aun asi`, `al final`, `con todo`,
  `a fin de cuentas`, `pero`, `sin embargo`...), no por dónde está.
- Robusto: ≈ 0,80 en todos los escenarios de la prueba de robustez. Pero en los datos originales queda 9 puntos
  por debajo del experimento 2.
- **Techo de información:** solo el 31 % de las reseñas negativo/positivo tiene un conector conclusivo. En el resto,
  el veredicto es la **última opinión sin marca**: "A excelente. B pésima." (negativo) y "B pésima. A excelente."
  (positivo) tienen las mismas palabras. Sin usar el orden no se pueden distinguir.
- Matiz: la prueba de robustez le favorece un poco, porque conserva el conector aunque la oración quede al inicio. En
  un caso realista (veredicto al inicio **sin** conector) el experimento 3 no se invierte, pero rinde como la línea base.
- **N-gramas más largos no ayudan** (texto completo, CV): (1,1) 0,739 · (1,2) 0,736 · (1,3) 0,728 · (1,6) 0,717.
  El orden relevante es de largo alcance (10–30 palabras) y los n-gramas largos son casi únicos.

### Experimento 4 — Última oración con opinión
- **Análisis de errores del experimento 2:** los errores se concentran en reseñas cuya última oración **no tiene el
  veredicto**:
  - **frases evasivas** (`cada quien que...`, `todavía no me decido`, `ni lo mejor ni lo peor`, `sigo con dudas`,
    `hay de todo`, `para gustos, colores`): ≈ 50 % de error;
  - **oraciones descriptivas** (`lo tengo en la sala`, `la garantía es de un año`): el veredicto está antes.
- **Ruido irreducible:** el **16 %** de las reseñas negativo/positivo termina en una evasiva, y en ellas **ninguna
  parte del texto predice la etiqueta mejor que el azar** (0,46–0,54 con texto completo, primera oración, última
  antes de la evasiva...). Esas etiquetas son prácticamente aleatorias. Consecuencia: **techo de accuracy ≈ 0,94**
  para cualquier modelo, y bastante ruido en el leaderboard (la diferencia 0,88 / 0,90 entre equipos puede ser en
  parte ruido).
- **Detector de opinión:** un TF-IDF + LR a nivel de **oración** que decide si una oración expresa opinión. Se entrena
  con oraciones de reseñas `neutral` (sin opinión) y oraciones de reseñas negativo/positivo que empiezan con un
  conector (con opinión), **dentro del pipeline** (clase `PartesConOpinion`), así que en cada fold solo ve datos de
  entrenamiento. Recorriendo la reseña desde el final, se elige la primera oración que no es evasiva y que el
  detector considera con opinión: la **última oración con opinión**. Cambia la oración elegida en el 24 % de las
  reseñas negativo/positivo.
- **De dónde viene la mejora (ablación, CV):**

  | Paso | Acc. CV | Aporte |
  |---|---|---|
  | Experimento 2 | 0,8865 | — |
  | + saltar evasivas (sin detector) | 0,8878 | +0,1 |
  | + **detector de opinión** (LR, `C = 10`) | 0,8964 | **+0,9** |
  | + ajustar `C` de LR (`C = 1`) | 0,8973 | +0,1 |
  | + LinearSVC (`C = 0,2`) | 0,9009 | +0,4 (dentro del ruido) |

  **La mejora viene de la representación (el detector), no del clasificador.** Regresión logística y LinearSVC,
  cada uno con su grilla de `C`, rinden prácticamente igual (0,8973 frente a 0,9009).
- Por grupos (CV): negativo/positivo **sin** evasiva final 0,904 → **0,927**; **con** evasiva 0,507 → 0,505 (azar,
  sin cambio); negativo/positivo predichas como `neutral` 84 → **29**.
- Gana al experimento 2 en los **5 folds**, pero en el test la diferencia no se confirma (0,8975 frente a 0,8992).
- Sigue dependiendo de la posición (robustez 0,736 / 0,595).
- **Debilidad del detector:** como sus ejemplos "con opinión" casi siempre empiezan con un conector, a veces confunde
  "empieza con `la verdad`" con "tiene opinión" y elige una oración descriptiva.

### Decisión
**Competir con la base del experimento 4** y reportar la robustez en cada experimento siguiente. Los experimentos 2 y
3 quedan como evidencia del compromiso entre accuracy y robustez para la sustentación, y motivan la Parte 2 (modelos
secuenciales).

### Descartes documentados
| Descartado | Motivo |
|---|---|
| KNN | Alta dimensión: las distancias se concentran; no aprende pesos con la etiqueta (TF-IDF pondera por rareza, no por relevancia). |
| Random Forest | Cada predicción consulta ≈ 14 features y el sentimiento es una suma de muchas pistas débiles; muestrear features en datos dispersos rinde mal. |
| SGDClassifier | Mismo modelo lineal con un optimizador aproximado; no aporta frente a LR y LinearSVC con 12.000 reseñas. |
| MLPClassifier | Es una red neuronal: **prohibida** en la Parte 1. |
| Lematización con spaCy, embeddings | Los modelos de spaCy y los embeddings son redes neuronales: prohibidos. |
| Quitar stopwords con listas estándar | Eliminan `no`, `pero`, `aunque`, `nada`. |
| L1 como regularización por defecto | Se queda con una feature por grupo de correlacionadas; en texto rinde igual o peor que L2. Se puede probar como variante. |
| N-gramas de palabras largos (1,3)+ en el texto completo | Empeoran (experimento 3): son casi únicos y no capturan el orden de largo alcance. |
| N-gramas de caracteres en la última oración con opinión | Probados en el experimento 4: no aportan (0,898 frente a 0,899 sin ellos). |
| Intentar predecir las reseñas que terminan en evasiva | Su etiqueta es prácticamente aleatoria (experimento 4, sección 2.2). |

---

## 6. Margen de acción y próximos experimentos (sobre la base del experimento 4)

### ¿Cuánto se puede mejorar?

| | Accuracy |
|---|---|
| Actual (CV, experimento 4) | ≈ 0,90 |
| Techo estimado (16 % de negativo/positivo con evasiva final, etiquetas aleatorias) | ≈ 0,94 |
| **Margen corregible** | **≈ 4 puntos**: el 7,3 % de error en reseñas negativo/positivo **sin** evasiva final |

Cada idea nueva tiene que atacar esos ≈ 4 puntos. Cambiar de clasificador o afinar `C` aporta décimas; lo que mueve
la accuracy es **qué información recibe cada parte del modelo**.

### Cómo buscar mejoras (método)
1. **Análisis de errores** con `cross_val_predict` sobre `X_train` del mejor modelo → buscar un patrón con tasa de
   error alta (inicio de la última oración, largo, conector, grupo).
2. **Hipótesis concreta** sobre por qué falla ese patrón.
3. **Ablación:** un cambio a la vez, midiendo cuánto aporta.
4. **Verificar por grupo y fold por fold:** la mejora debe aparecer en el grupo que se quería corregir.
5. Al final, ajustar clasificador y `C` con CV (ahí casi no está la ganancia).

### Experimentos propuestos

Máximo ≈ 10 envíos en total (van 4). Se envía solo si la mejora en CV supera el ruido entre folds.

| # | Experimento | Qué probar | Por qué |
|---|---|---|---|
| 5 | **Detector de opinión mejorado** | Quitar el conector inicial de la oración antes de pasarla al detector (que aprenda del contenido, no del conector); ajustar el umbral con CV (`partes__umbral` en la grilla); ejemplos "con opinión" más variados | El detector es lo que más aportó y tiene un sesgo conocido hacia oraciones que empiezan con conector |
| 6 | **Tipo de conector + opinión anterior** | Agregar a la última oración con opinión un token con su tipo de conector (`TIPO_conclusivo`, `TIPO_concesivo`, `TIPO_eso_si`); darle su propio TF-IDF a la **penúltima oración con opinión** | Un veredicto introducido por "aun así" pesa distinto que uno introducido por "de paso"; las reseñas difíciles tienen dos opiniones de signo contrario |
| 7 | **Representación por parte + negación** | `min_df`, n-gramas (1,3) solo en la última oración con opinión, binario frente a TF-IDF; marcar la negación (`no_funciona`) dentro de esa oración; stemming | Ajuste fino de la parte que más pesa |
| 8 | **Otros clasificadores** | MultinomialNB / ComplementNB con su grilla de `alpha`, en el mismo `GridSearchCV` que LR y LinearSVC | Completar la comparación de modelos (esperable: no supera a los lineales) |
| 9 | **Ajuste fino conjunto / ensamble** | Grilla fina alrededor del mejor modelo; votación LR + LinearSVC; L1 / Elastic Net | Últimos décimos |
| 10 | Reserva | Solo si el análisis de errores sugiere algo concreto | — |

---

## 7. Pendientes

- [ ] **Subir a Kaggle** `submission_01`, `03` y `04.csv` y registrar el score público (tabla de la sección 1 y
      columna `kaggle_publico`). Mínimo **5 envíos distintos** para que cuente la participación.
- [ ] Confirmar que el 0,88 público corresponde a `submission_02.csv`.
- [ ] Agregar en el análisis del experimento 3 el matiz sobre la prueba de robustez y la tabla de n-gramas largos.
- [ ] Agregar en cada notebook una celda que imprima las versiones de las librerías, y un `requirements.txt`.
- [ ] Al terminar: **consolidar todo en un único notebook final** (la entrega exige uno solo), con una tabla resumen de
      todos los experimentos, y guardar el `.joblib` del **mejor envío del leaderboard público**.
- [ ] Antes de la entrega: meter la normalización dentro del pipeline del modelo final, para que el `.joblib` reciba
      texto crudo.
- [ ] Confirmar con el profesor que LinearSVC (SVM) cuenta como modelo "visto en la primera parte del curso" (si no,
      la regresión logística rinde prácticamente igual en el experimento 4).

### Fechas
- Cierre de la competencia de la Parte 1: **semana 9**.
- Entrega en Bloque Neón (notebook + modelo `.joblib`/`pickle` del mejor envío): **semana 11**.
