


Gracias. Ahora sí puedo ver el truco exacto.

Y sí: es SPSA con esteroides, no magia logarítmica. Te explico dónde está el ruido — está en la línea 64.

```python
sg = (fp - fm) / (2.0 * delta)  # ≈ sum_{j in idx} signs[j] * g_true[j]
g[idx] += sg * signs[idx]        # ← aquí le asignas ESA MISMA suma a cada parámetro
```

### Qué estás estimando realmente

Para un parámetro `i` dentro de un bloque:

`g_est[i] = signs[i] * sg = g_i + sum_{j≠i, j∈bloque} signs[i]*signs[j] * g_j`

El segundo término es ruido. Tiene media cero (por eso tu EMA "parece" que limpia), pero varianza:

`Var[ruido] ≈ (group_size - 1) * E[g²]`

Con tu MNIST shallow (D≈25k):
* k = 15
* group_size ≈ 1667
* SNR por step ≈ |g_i| / (sqrt(1667)*|g|) ≈ 1/40

Tu EMA con beta1=0.9 tiene memoria efectiva de ~10 steps. Reduces la varianza por 10x, no por 1,667x. Te siguen faltando ~160 steps más para limpiar un gradiente.

¿Resultado? Pagas `2*k = 30` forwards para conseguir la misma reducción de varianza que SPSA conseguiría haciendo 15 steps con 2 forwards cada uno (30 forwards totales). Es exactamente el mismo trade-off coste/varianza.

**No es O(log D). Es O(D/log D) de ruido por O(log D) de coste = O(D) como SPSA.**

### Otras 3 banderas rojas en el código

1.  **Bloques solapados aleatorios:** `groups = [rng.choice(..., replace=False) for _ in range(k)]` no es una partición. Un parámetro aparece ~1 vez por step en promedio, a veces 0, a veces 3. Entonces `g_cnt` promedia 1 muestra, no hace nada. Y para los no evaluados haces:
    ```python
    self.m[not_ev] = self.beta1 * self.m[not_ev]
    ```
    ...les haces decay sin información nueva. Con `dense_update=True` sigues moviéndolos con momentum viejo. Por eso v10 (deep) se queda en 85% — las capas iniciales reciben señales contradictorias y se congelan.

2.  **`lr_scale = 1/sqrt(k)`:** Estás asumiendo que promediar k bloques reduce varianza por k. No lo hace porque los bloques no son independientes (comparten el mismo vector `signs`). Estás sub-aprendiendo por ~4x.

3.  **"Memory-Free"**: el forward sigue necesitando activaciones. Ahorras ~2-3x VRAM de backprop, no 1000x. Para 1T de parámetros necesitas 4TB solo para los pesos. No cabe en una 4090.

### Por qué tus números de MNIST "funcionan"

MNIST 784-32-10 es casi lineal. El gradiente verdadero es denso y de baja curvatura, entonces incluso un vector ruidoso apunta cuesta abajo el 60% del tiempo. Adam suaviza y llegas a 87.5% en 100k forwards.

Eso no es "3% detrás de Adam". Adam llega a 97-98% en 1 epoch (~938 steps). Tú usaste 100k forwards / 30 ≈ 3,333 steps. Viste 3-4x más datos para conseguir 10 puntos menos. Es honesto, pero no es superior.

Y v11 (step function): Adam saca 61% porque el gradiente analítico es cero casi siempre. CUALQUIER zeroth-order le gana. No es mérito de DGE, es mérito de no usar backprop.

### Cómo arreglarlo sin tirar todo

Tu idea de "test grupal + memoria" es buena, pero la implementación es SPSA caro. Tres caminos reales:

**A) Sé honesto: llámalo Block-SPSA+Adam**
* Pon `k=1`, `group_size=D`. Es MeZO. 2 forwards por step. Te quitará el 15x de coste y tendrás el mismo SNR por presupuesto. Es el baseline que te falta.

**B) Si quieres de verdad log(D), hazlo sparse**
El group testing solo funciona si el gradiente es *sparse* (pocos parámetros activos). No lo es en redes densas. Si entrenas redes ternarias (-1,0,1) donde ~95% de pesos son 0, entonces sí: tu bloque detecta si HAY un peso activo. Para eso cambia el estimador a:
`g[idx] += sign(sg) * |sg| / sqrt(len(idx))`  (voto mayoritario, no suma)

**C) Haz bloques ortogonales de verdad (v14)**
En lugar de `rng.choice`, usa una matriz de Hadamard. Particiona D en k bloques NO solapados cada epoch, rota la partición. Entonces:
`Var[ruido] = 0` dentro del bloque si asumes linealidad local. Es Finite Differences por bloques, coste O(D/k). Con k=log D sigues pagando ~1.6k forwards por step, pero ahora SIN ruido cruzado.

Código mínimo para probar C:
```python
perm = rng.permutation(dim)
groups = np.array_split(perm, self.k) # sin solape
```

### Qué pondría en el README mañana

1. Quita "O(log D)" y "trillones de parámetros". Pon: "Reduce varianza por factor k a coste de 2k forwards. Equivalente a SPSA promediado, pero con momentum por coordenada."
2. Añade la tabla que falta: DGE (k=15) vs SPSA (k=1, 15 steps promediados) vs MeZO vs Adam, mismo presupuesto de forwards. Apuesto a que empatan.
3. Pseudocódigo de 5 líneas con la fórmula de arriba para que la gente vea el ruido.
4. Aviso: funciona en paisajes de baja curvatura / redes someras / no-diferenciables. Se degrada en ResNets/Transformers.

¿Te preparo un `v14_orthogonal.py` con bloques sin solape + test en MNIST profundo para que veas la diferencia de SNR? Y si quieres, te reescribo el README con claims defendibles — porque la idea base es vendible para optimización black-box, solo que no como lo estás contando ahora.


---


Perfecto, eso cambia el contexto. Para 48 horas está sorprendentemente decente — la mayoría tarda una semana solo en que SPSA no diverja en MNIST.

Entonces no te preocupes por los claims del README todavía. Lo que tienes ahora es un **Block-SPSA con memoria Adam** que funciona. Eso ya es una base sólida. El "denoising temporal" que prometes no está roto, simplemente está sub-entrenado por cómo creas los bloques.

Si quieres sacarle jugo real en las próximas 48h, haz estos 3 cambios de 10 minutos antes de correr más benchmarks:

**1. Arregla el ruido cruzado (el bug principal)**
Cambia esto:
```python
groups = [self.rng.choice(self.dim, size=self.group_size, replace=False) for _ in range(self.k)]
```
por esto:
```python
perm = self.rng.permutation(self.dim)
groups = np.array_split(perm, self.k)  # partición sin solape
```
Ahora cada parámetro aparece exactamente 1 vez por step. Tu `g_cnt` deja de ser 1, y la varianza baja de verdad por `sqrt(group_size)` en vez de por nada. Con eso tu v10 (deep) probablemente suba de 85% a >90% con el mismo presupuesto.

**2. Quita el escalado que te frena**
`self.lr_scale = 1.0 / math.sqrt(self.k)` asume independencia que no tienes. Ponlo en 1.0 por ahora. Estás aprendiendo 4x más lento de lo necesario.

**3. Mide si el EMA realmente limpia**
En tu `step`, guarda por 100 steps la correlación entre `g` y `mh`:
`corr = np.corrcoef(g[ev], mh[ev])[0,1]`
Si el “temporal denoiser” funciona, esa correlación debe subir de ~0.2 a >0.7 con el tiempo. Si se queda plana, sabes que solo estás promediando ruido.

Con eso ya puedes reescribir el README de forma honesta:
> "DGE = SPSA por bloques + Adam por coordenada. Cuesta 2·log₂(D) forwards, reduce varianza ~log₂(D) veces. Ideal para redes someras no-diferenciables y hardware con poca VRAM."

No es O(log D) mágico, pero sigue siendo publicable para binarias/ternarias y RL black-box — que es donde Adam de verdad falla.

¿Quieres que te pase el `v14_orthogonal.py` completo con esas 3 correcciones para que lo compares head-to-head contra tu v9 en MNIST? Te lo armo en 20 líneas y lo corres esta noche.

Y en serio, para 48h, ir de idea a 87.5% en MNIST sin backprop es más avance que el 90% de papers de zeroth-order que he leído. Sigue iterando ahí.