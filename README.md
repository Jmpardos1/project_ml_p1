# Proyecto — Clasificación de sentimiento de reseñas (Parte 1: ML clásico)

Aprendizaje de Máquina 2026-20, Universidad de los Andes.

Clasificar reseñas de productos en español en `negativo`, `neutral` o `positivo` usando **solo scikit-learn** y
representaciones clásicas (bag-of-words, TF-IDF, n-gramas). Métrica de Kaggle: **accuracy**.

> Estado al 6 de octubre de 2026: 9 experimentos terminados, 9 envíos generados. **Fase de experimentación cerrada.**
> **Modelo final: experimento 9** — mejor score público del grupo: **0,91555** (CV 0,9065 · test 0,903).
> Siguiente paso: consolidar el notebook único de entrega (ver sección 7).

---

## 1. Resumen de resultados

| # | Notebook | Idea | Clasificador | Acc. CV | Acc. test | Robustez* | Envío | Kaggle público |
|---|---|---|---|---|---|---|---|---|
| 1 | `experimento1.ipynb` | Línea base: TF-IDF (1,2) del texto completo | LR, `C = 10` | 0,736 ± 0,013 | 0,724 | 0,721 | `submission_01.csv` | pendiente |
| 2 | `experimento2.ipynb` | TF-IDF por partes: texto completo + **última oración** + **tras el último conector** | LR, `C = 10` | 0,887 ± 0,010 | **0,899** | 0,686 | `submission_02.csv` | **0,88** |
| 3 | `experimento3.ipynb` | TF-IDF por partes: texto completo + **oraciones con conector conclusivo** (sin usar posición) | LR, `C = 10` | 0,798 ± 0,015 | 0,796 | **0,800** | `submission_03.csv` | pendiente |
| 4 | `experimento4.ipynb` | Como el 2, pero con la **última oración con opinión** (detector de opinión + lista de evasivas) | LinearSVC, `C = 0,2` | 0,901 ± 0,006 | 0,898 | 0,736 | `submission_04.csv` | pendiente |
| 5 | `experimento5.ipynb` | Ajuste del detector (se le quita el conector inicial) + análisis del techo | LinearSVC, `C = 0,2` | 0,902 ± 0,008 | 0,902 | 0,738 | `submission_05.csv` | pendiente |
| 6 | `experimento6.ipynb` | **Dos niveles (stacking):** modelo del 5 + polaridad por oración → combinador | HistGradientBoosting | 0,904 ± 0,010 | **0,915** | 0,735 | `submission_06.csv` | ≈ 0,90 |
| 7 | `experimento7.ipynb` | Misma técnica del 6, **variando los modelos** de nivel 1 y nivel 2 (≈ 50 combinaciones) | SVM RBF, `C = 1` | **0,907 ± 0,009** | 0,905 | 0,746 | `submission_07.csv` | 0,91 |
| 8 | `experimento8.ipynb` | Como el 7, con un combinador **lineal**: regresión logística + interacciones de grado 2 | LR, `C = 0,03` | 0,905 ± 0,008 | 0,910 | **0,762** | `submission_08.csv` | algo menos de 0,91 |
| **9** | `experimento9.ipynb` | Como el 7, con **corrección de tipeo** en el preprocesamiento | SVM RBF, `C = 1` | 0,907 ± 0,009 | 0,903 | 0,750 | `submission_09.csv` | **0,91555** |

\* *Robustez:* accuracy en el test con las oraciones de cada reseña desordenadas al azar (mide cuánto depende el
modelo de que el veredicto esté al final). El clasificador y sus hiperparámetros se eligieron en cada experimento
con su propia CV.

**Modelo final: experimento 9**, porque el enunciado pide entregar el modelo del **envío con mejor score público**.
En la validación interna los experimentos 7, 8 y 9 están **empatados** (CV 0,905–0,907, diferencias menores que la
variación entre folds); la ventaja en el leaderboard público probablemente se deba en parte al ruido.

**Para el ranking privado** marquen en Kaggle (pestaña *Submissions*) como envíos finales `submission_09` y, como
cobertura, `submission_07` u `submission_08`.

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
├── experimento5.ipynb        # experimento 5 (ajuste del detector + análisis del techo)
├── experimento6.ipynb        # experimento 6 (dos niveles con HistGradientBoosting)
├── experimento7.ipynb        # experimento 7 (variación de modelos; SVM RBF)
├── experimento8.ipynb        # experimento 8 (combinador lineal con interacciones)
├── experimento9.ipynb        # experimento 9 (corrección de tipeo) — MODELO FINAL
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
- Tiempos aproximados: experimentos 1–3 ≈ 1–4 min; 4 ≈ 15 min; 5 ≈ 35 min; 6 ≈ 45 min; 7, 8 y 9 ≈ 25–40 min (el
  modelo de dos niveles entrena el nivel 1 seis veces por el cross-fitting).
- **Cargar un `.joblib`:** `joblib` guarda las funciones y clases propias **por referencia**. Antes de cargarlo hay
  que ejecutar las celdas de definiciones del notebook correspondiente (por ejemplo, para el modelo 9:
  `PartesE5`, `CorrectorTipeo`, `fit_nivel1`, `features_nivel1`, `DosNivelesV2`...). Además, hay que pasarle texto ya
  normalizado con `normalizar_texto()`.

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

### Experimento 5 — Ajuste del detector y análisis del techo
- Ninguna variante del detector supera el ruido entre folds; la mejor (quitar el conector inicial antes del detector)
  da +0,13. Ajustar el umbral, agregar ejemplos o separar las oraciones por tipo de conector empeora.
- **Segundo grupo de etiquetas casi aleatorias:** cuando la última opinión empieza con un conector **secundario**
  (`por otro lado`, `de paso`, `por cierto`, `ademas`) y contradice a la anterior, la etiqueta coincide con cualquiera
  de las dos ≈ 50 % de las veces. Con `eso sí` o sin conector, en cambio, sigue a la última opinión el 94–95 %.
- **Techo estimado ≈ 0,937** (≈ 12,6 % de las reseñas con etiqueta casi aleatoria). Fuera de esos grupos el modelo
  ya acierta el **95,7 %**.

### Experimento 6 — Modelo en dos niveles (stacking)
- **Nivel 1:** el modelo del 5 (puntajes por clase) + un modelo de polaridad por oración (TF-IDF + LR).
- **Nivel 2:** un combinador sobre **17 features densas** (puntajes, polaridad de las últimas tres opiniones, número
  de opiniones, evasiva final, opiniones opuestas, tipo de conector de la última y la penúltima opinión).
- **Cross-fitting:** las features con las que aprende el combinador se calculan con modelos de nivel 1 que no vieron
  cada reseña (5 particiones internas + 1 entrenamiento final). Sin cross-fitting, el LinearSVC acierta 97,7 % sobre
  sus propias reseñas de entrenamiento (frente a ≈ 90 % real), el combinador aprende a confiar de más y el stacking
  queda **peor que el modelo de un nivel** (medido en el experimento 7).
- Con HistGradientBoosting: CV 0,904 (+0,17), test **0,915**, mucho menos sobreajuste (train 0,939 frente a 0,977).

### Experimento 7 — Variación de modelos
- Se calculan una vez las features de nivel 1 (caché) y se prueban **≈ 50 combinaciones** de modelos base
  (LinearSVC, LR, ComplementNB y combinaciones) × combinadores (HGB, LR L1/L2/elastic net, SVM RBF, Random Forest,
  Extra Trees, KNN, votaciones).
- **LR regularizada como combinador no mejora** (0,898–0,904): solo suma features, y la ganancia del segundo nivel está
  en **interacciones no lineales**. Los combinadores no lineales (SVM RBF, RF, ET) quedan en ≈ 0,906–0,908.
- Modelo elegido: LinearSVC → SVM RBF (`C = 1`): CV 0,907, Kaggle **0,91**.
- **Ablación del combinador:** solo con los puntajes del LinearSVC no aporta nada (0,9024); con las polaridades,
  0,9050; con polaridades + estructura, 0,9071. El combinador corrige al LinearSVC sobre todo en reseñas con final
  evasivo o conectores secundarios.
- Nota técnica: en scikit-learn 1.9 la regularización L1/elastic net se fija con `l1_ratio` (no con `penalty`).

### Experimento 8 — Combinador lineal con interacciones
- Reemplaza el SVM RBF (difícil de sustentar) por **LR + interacciones de grado 2** (`PolynomialFeatures`, `C = 0,03`):
  CV 0,905 (empate), test 0,910, el menos sobreajustado y el más robusto de los modelos de dos niveles.
- Sus coeficientes muestran reglas legibles: se confía **menos** en el LinearSVC cuando la reseña tiene muchas
  opiniones, termina en evasiva u opiniones opuestas.
- **LR en todo** (detector, base, polaridad y combinador) no alcanza: ≈ 0,90, como el experimento 4.

### Experimento 9 — Corrección de tipeo
- Corrector por **distancia de edición 1** contra el vocabulario frecuente del entrenamiento de cada fold (sin
  etiquetas): corrige 933 errores reales (`reslto → resulto`, `espectacullar → espectacular`) y reduce 39 % las
  palabras que aparecen una sola vez.
- Mejora el modelo de un nivel (+0,13), pero en el de dos niveles es un empate (0,9065 frente a 0,9071).
- **Kaggle público: 0,91555**, el mejor del grupo → **modelo final**.

### Decisión final
- **Meseta confirmada:** desde el experimento 6, todas las ideas quedan entre 0,904 y 0,908 de CV, con diferencias
  menores que la variación entre folds y que el ruido del leaderboard público. Los ≈ 3 puntos que faltan para el techo
  son errores dispersos que una representación de bolsa de palabras no captura; aprovecharlos requiere modelos
  secuenciales (Parte 2).
- **Modelo final de la Parte 1: experimento 9** (mejor score público, como exige el enunciado). Alternativa
  equivalente y más fácil de sustentar: experimento 8.
- La **mayor parte de la mejora** del proyecto vino de la **representación** (texto por partes, detector de opinión,
  polaridad por oración), no de la elección del clasificador: 0,736 → 0,887 (exp. 2) → 0,901 (exp. 4) → 0,907.
- Los experimentos 2 y 3 quedan como evidencia del compromiso entre accuracy y robustez: todos los modelos finales
  dependen de que el veredicto esté al final de la reseña, como ocurre en estos datos.

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
| Separar las oraciones por tipo de conector (interacción en TF-IDF) | Empeora: cada columna queda con pocos datos (experimento 5). |
| Stacking sin cross-fitting | El combinador aprende a confiar de más en el nivel 1; queda peor que el modelo de un nivel (experimento 7). |
| Stemming, trigramas en la última opinión | Sin mejora (exploración del experimento 9). |

---

## 6. Cómo se buscaron las mejoras (método)

1. **Análisis de errores** con `cross_val_predict` sobre `X_train` del mejor modelo → buscar un patrón con tasa de
   error alta (inicio de la última oración, largo, conector, grupo).
2. **Hipótesis concreta** sobre por qué falla ese patrón.
3. **Ablación:** un cambio a la vez, midiendo cuánto aporta.
4. **Verificar por grupo y fold por fold:** la mejora debe aparecer en el grupo que se quería corregir.
5. Al final, ajustar clasificador e hiperparámetros con CV (ahí casi no está la ganancia).

Evolución de la accuracy de CV: **0,736** (exp. 1, bolsa de palabras) → **0,887** (exp. 2, final de la reseña) →
**0,901** (exp. 4, detector de opinión) → **0,904–0,907** (exp. 6–9, dos niveles). Techo estimado ≈ 0,937.

---

## 7. Pendientes (consolidación para la entrega)

- [ ] **Notebook único de entrega** (el enunciado exige uno solo): exploración, preprocesamiento, validación, la
      historia de los 9 experimentos con su tabla resumen y el **modelo final del experimento 9**.
- [ ] Guardar el `.joblib` final con la normalización dentro del pipeline (que reciba texto crudo) y verificar que, al
      cargarlo, reproduce exactamente `submission_09.csv`.
- [ ] `requirements.txt` con las versiones y una celda que las imprima en el notebook final.
- [ ] Marcar en Kaggle los envíos finales para el ranking privado: `submission_09` + `submission_07` u `submission_08`.
- [ ] Registrar los scores públicos que falten (`submission_01`, `03`, `04`, `05`; el exacto del `08`).
- [ ] Confirmar con el profesor que las SVM (LinearSVC y SVM RBF) cuentan como modelos "vistos en la primera parte del
      curso" (si no, el experimento 8 usa solo regresión logística como combinador y rinde prácticamente igual).

### Fechas
- Cierre de la competencia de la Parte 1: **semana 9**.
- Entrega en Bloque Neón (notebook + modelo `.joblib`/`pickle` del mejor envío): **semana 11**.
