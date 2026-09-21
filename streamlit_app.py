"""
¿Cuánto debería costar mi crédito?
Referencia de tasas de crédito empresarial para MiPymes en Colombia.

Proyecto de Aplicación 1 — Datos Abiertos
Técnicas de Aprendizaje de Máquina · Pontificia Universidad Javeriana
Nameer Chowdhury · Mariana Rodríguez
"""

import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="¿Cuánto debería costar mi crédito?",
                   page_icon="💰", layout="wide")


# ----------------------------------------------------------------------------
# Carga de modelos (cacheada: solo ocurre una vez)
# ----------------------------------------------------------------------------
@st.cache_resource
def cargar():
    return (joblib.load("modelo_mercado.joblib"),
            joblib.load("modelo_riesgo.joblib"),
            joblib.load("config_app.joblib"))


M_MERCADO, M_RIESGO, CFG = cargar()

RIESGO_CAT = CFG["RIESGO_CAT"]
RIESGO_NUM = CFG["RIESGO_NUM"]
PROTEGIDAS = CFG["PROTEGIDAS"]
COLS_MERCADO = RIESGO_CAT + PROTEGIDAS + RIESGO_NUM
COLS_RIESGO = RIESGO_CAT + RIESGO_NUM

TOPES = CFG["topes_actuales"]
MES_TOPES = CFG["mes_topes"]
MES_MODELO = CFG["mes_modelo"]
MAE_MOD = CFG["mae_modalidad"]
ENT_POR_MOD = CFG["entidades_por_modalidad"]
PROD_POR_MOD = CFG["productos_por_modalidad"]
TIPO_ENT = CFG["tipo_por_entidad"]
OPC = CFG["opciones"]

# Salario mínimo usado para ubicar el monto en su banda. Actualizar cada enero.
SMLMV = 1_623_500

# ----------------------------------------------------------------------------
# Etiquetas legibles
# ----------------------------------------------------------------------------
MODALIDADES = {
    "CREDITO POPULAR PRODUCTIVO URBANO":
        "Crédito popular productivo urbano — microempresa en ciudad",
    "CREDITO POPULAR PRODUCTIVO RURAL":
        "Crédito popular productivo rural — microempresa en zona rural",
    "CREDITO PRODUCTIVO URBANO":
        "Crédito productivo urbano — pequeña empresa en ciudad",
    "CREDITO PRODUCTIVO RURAL":
        "Crédito productivo rural — pequeña empresa en zona rural",
    "CREDITO PRODUCTIVO MAYOR MONTO": "Crédito productivo de mayor monto",
    "CONSUMO Y ORDINARIO": "Crédito comercial ordinario — empresa consolidada",
}
MODALIDAD_INV = {v: k for k, v in MODALIDADES.items()}

TASAS = {
    "FS": "Tasa fija — la cuota no cambia nunca",
    "IB1": "Tasa variable atada al IBR a 1 mes",
    "IB3": "Tasa variable atada al IBR a 3 meses",
    "IB6": "Tasa variable atada al IBR a 6 meses",
    "IBR": "Tasa variable atada al IBR",
    "DTF": "Tasa variable atada a la DTF",
    "DTFE": "Tasa variable atada a la DTF efectiva",
}
TASA_INV = {v: k for k, v in TASAS.items()}

GARANTIAS = {
    "Sin garantia": "Sin garantía",
    "Garantia idónea o no idónea": "Garantía propia (hipoteca, prenda, codeudor)",
    "Garantía fondo nacional de garantías (FNG)": "Garantía del FNG",
    "Garantía del fondo agropecuario de garantías (FAG)":
        "Garantía del FAG (agropecuario)",
    "Garantía fondo de garantías de Antioquia (FGA)":
        "Garantía del FGA (Antioquia)",
    "Garantía  fondo nacional de garantías (FNG) o Fondo de Garantías de Antioquia (FGA)":
        "Garantía del FNG o del FGA",
}
GARANTIA_INV = {v: k for k, v in GARANTIAS.items()}

SECTORES = {
    "Comercio al por menor (tienda, minimercado, papelería)": "47",
    "Comercio al por mayor / distribución": "46",
    "Restaurantes, comidas y bebidas": "56",
    "Agricultura y ganadería": "01",
    "Confecciones y textiles": "14",
    "Elaboración de alimentos": "10",
    "Construcción de edificios y obras": "41",
    "Transporte de carga o pasajeros": "49",
    "Peluquería y servicios personales": "96",
    "Servicios informáticos y software": "62",
    "Salud": "86", "Educación": "85", "Otro sector": "47",
}

DEPARTAMENTOS = {
    "Bogotá D.C.": "11", "Antioquia": "05", "Valle del Cauca": "76",
    "Atlántico": "08", "Santander": "68", "Cundinamarca": "25",
    "Bolívar": "13", "Nariño": "52", "Córdoba": "23", "Tolima": "73",
    "Huila": "41", "Boyacá": "15", "Norte de Santander": "54",
    "Cauca": "19", "Meta": "50", "Magdalena": "47", "Caldas": "17",
    "Risaralda": "66", "Cesar": "20", "Quindío": "63", "Sucre": "70",
    "Casanare": "85", "La Guajira": "44", "Putumayo": "86",
    "Chocó": "27", "Caquetá": "18", "Arauca": "81", "Otro": "11",
}

MESES_PLAZO = {
    "Hasta 30 días": 1, "Más de 30 días y hasta 1 año": 12,
    "Más de 1 año y hasta 3 años": 24, "Más de 3 años y hasta 5 años": 48,
    "Más de 5 años y hasta 7 años": 72, "A más de 7 años": 96,
}

ETNIA_DEFAULT = "Sin información (1)"


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def banda_monto(monto_cop):
    """Ubica el monto en la banda de SMLMV más estrecha que lo contiene."""
    smlmv = monto_cop / SMLMV
    mejor, ancho_min = None, float("inf")
    for op in OPC["rango_monto_desembolsado"]:
        nums = [float(x.replace(",", "."))
                for x in re.findall(r"(\d+(?:[.,]\d+)?)\s*SMLMV", op)]
        if not nums:
            continue
        if op.lower().startswith("hasta"):
            lo, hi, abierta = 0.0, nums[0], False
        elif len(nums) >= 2:
            lo, hi, abierta = nums[0], nums[1], False
        else:
            lo, hi, abierta = nums[0], float("inf"), True
        if (lo < smlmv <= hi) or (lo == 0 and smlmv <= hi):
            # Las bandas cerradas siempre ganan sobre la abierta
            ancho = (hi - lo) if not abierta else float("inf")
            if ancho < ancho_min or (mejor is None):
                mejor, ancho_min = op, ancho
    if mejor is None:
        mejor = max(OPC["rango_monto_desembolsado"],
                    key=lambda o: max([float(x) for x in re.findall(r"(\d+)", o)] or [0]))
    return mejor


def fila(modalidad, producto, garantia, plazo, banda, tasa, tamano, antiguedad,
         persona, deudor, sector, entidad, monto, sexo, depto):
    return {
        "modalidad_tibc": modalidad,
        "producto_de_cr_dito": producto,
        "tipo_de_cr_dito": ("Crédito productivo"
                            if modalidad != "CONSUMO Y ORDINARIO"
                            else "Comercial ordinario"),
        "tipo_de_garant_a": garantia,
        "plazo_de_cr_dito": plazo,
        "rango_monto_desembolsado": banda,
        "tipo_de_tasa": tasa,
        "tama_o_de_empresa": tamano,
        "antiguedad_de_la_empresa": antiguedad,
        "tipo_de_persona": persona,
        "clase_deudor": deudor,
        "ciiu_division": sector,
        "nombre_tipo_entidad": TIPO_ENT.get(entidad, "BC-ESTABLECIMIENTO BANCARIO"),
        "nombre_entidad": entidad,
        "mes": MES_MODELO,
        "sexo": sexo,
        "grupo_etnico": ETNIA_DEFAULT,
        "cod_departamento": depto,
        "log_monto_por_credito": float(np.log1p(max(monto, 1))),
    }


def predecir(d, modelo, cols):
    return float(modelo.predict(pd.DataFrame([d])[cols])[0])


def cuota_mensual(monto, tasa_ea, meses):
    i = (1 + tasa_ea / 100) ** (1 / 12) - 1
    if i <= 0 or meses <= 0:
        return monto / max(meses, 1)
    return monto * i / (1 - (1 + i) ** (-meses))


# ----------------------------------------------------------------------------
# Encabezado
# ----------------------------------------------------------------------------
st.title("¿Cuánto debería costar mi crédito?")
st.markdown("""
Si usted tiene una micro, pequeña o mediana empresa y va a pedir un crédito,
**no tiene cómo saber si la tasa que le ofrecen es razonable**. Los bancos sí:
conocen todo el mercado. Usted solo ve su cotización.

Esta herramienta usa los datos que todas las entidades vigiladas le reportan a la
Superintendencia Financiera para decirle tres cosas: qué tasa le cobrarían hoy a
una empresa como la suya, qué podría hacer para bajarla, y qué entidades han
cobrado menos por créditos parecidos.
""")

# ----------------------------------------------------------------------------
# Formulario
# ----------------------------------------------------------------------------
st.sidebar.header("Su crédito")

modalidad_lbl = st.sidebar.selectbox(
    "Tipo de crédito", list(MODALIDADES.values()),
    help="Si no sabe cuál le corresponde, elija según el tamaño de su empresa "
         "y si está en ciudad o zona rural")
modalidad = MODALIDAD_INV[modalidad_lbl]

prods = PROD_POR_MOD.get(modalidad, [])
producto = st.sidebar.selectbox("Producto específico", prods) if prods else None

monto = st.sidebar.number_input("Monto que necesita (COP)", min_value=100_000,
                                max_value=50_000_000_000, value=10_000_000,
                                step=1_000_000)
plazo = st.sidebar.selectbox("Plazo", list(MESES_PLAZO.keys()), index=2)
tasa_lbl = st.sidebar.selectbox("Tipo de tasa", list(TASAS.values()))
garantia_lbl = st.sidebar.selectbox("Garantía", list(GARANTIAS.values()))

st.sidebar.header("Su empresa")
tamano = st.sidebar.selectbox("Tamaño de la empresa", OPC["tama_o_de_empresa"],
                              index=OPC["tama_o_de_empresa"].index("Microempresa")
                              if "Microempresa" in OPC["tama_o_de_empresa"] else 0)
antiguedad = st.sidebar.selectbox("Antigüedad de la empresa",
                                  OPC["antiguedad_de_la_empresa"])
persona = st.sidebar.selectbox("¿Persona natural o jurídica?",
                               OPC["tipo_de_persona"])
deudor = st.sidebar.selectbox("¿Ya es cliente de esa entidad?",
                              OPC["clase_deudor"])
sector_lbl = st.sidebar.selectbox("Sector de su negocio", list(SECTORES.keys()))
depto_lbl = st.sidebar.selectbox("Departamento", list(DEPARTAMENTOS.keys()))

ents = ENT_POR_MOD.get(modalidad, [])
entidad = st.sidebar.selectbox("Entidad donde piensa pedirlo", ents) if ents else None

sexo = st.sidebar.selectbox(
    "Sexo del solicitante (opcional)", ["No aplica", "Femenino", "Masculino"],
    help="Solo se usa para comprobar si el mercado cobra distinto por este motivo")

calcular = st.sidebar.button("Calcular mi tasa de referencia",
                             type="primary", use_container_width=True)

# ----------------------------------------------------------------------------
# Resultado
# ----------------------------------------------------------------------------
if not calcular:
    st.info("Complete los datos en el panel de la izquierda y presione "
            "**Calcular mi tasa de referencia**.")
    st.stop()

if not producto or not entidad:
    st.error("No hay datos suficientes para esta combinación. "
             "Pruebe con otro tipo de crédito.")
    st.stop()

garantia = GARANTIA_INV[garantia_lbl]
tasa = TASA_INV[tasa_lbl]
sector = SECTORES[sector_lbl]
depto = DEPARTAMENTOS[depto_lbl]
banda = banda_monto(monto)
tope = TOPES[modalidad]

base = fila(modalidad, producto, garantia, plazo, banda, tasa, tamano,
            antiguedad, persona, deudor, sector, entidad, monto, sexo, depto)

r_mercado = predecir(base, M_MERCADO, COLS_MERCADO)
r_riesgo = predecir(base, M_RIESGO, COLS_RIESGO)
ea_mercado = min(r_mercado, 1.0) * tope
ea_riesgo = min(r_riesgo, 1.0) * tope

mae = MAE_MOD.get(modalidad, 2.0)
lo, hi = max(ea_mercado - mae, 0), min(ea_mercado + mae, tope)
meses = MESES_PLAZO.get(plazo, 24)
cuota = cuota_mensual(monto, ea_mercado, meses)

# --- Bloque 1: la cifra ------------------------------------------------------
st.header("Su estimación")

c1, c2, c3 = st.columns(3)
c1.metric("Tasa estimada", f"{ea_mercado:.1f} %",
          help="Tasa efectiva anual que el mercado cobra hoy a un perfil como el suyo")
c2.metric("Rango probable", f"{lo:.1f} – {hi:.1f} %")
c3.metric("Cuota mensual", f"${cuota:,.0f}", help=f"Durante {meses} meses")

pct = r_mercado * 100
if r_mercado >= 0.90:
    st.error(f"**Le están cobrando prácticamente el máximo que permite la ley.** "
             f"Su estimación es el {pct:.0f} % del tope de usura.")
elif r_mercado >= 0.75:
    st.warning(f"**Es una tasa alta** para este tipo de crédito: "
               f"el {pct:.0f} % del tope de usura.")
elif r_mercado >= 0.55:
    st.info(f"**Es una tasa intermedia**: el {pct:.0f} % del tope de usura.")
else:
    st.success(f"**Es una tasa favorable**: el {pct:.0f} % del tope de usura.")

st.caption(f"El tope legal de usura para esta modalidad es **{tope:.2f} %** "
           f"efectivo anual, certificado para {MES_TOPES}. Cobrar por encima "
           f"es el delito de usura del artículo 305 del Código Penal.")

# --- Bloque 2: mercado vs riesgo ---------------------------------------------
st.subheader("Lo que el mercado cobra vs. lo que justifica su crédito")

st.dataframe(pd.DataFrame({
    "": ["Lo que el mercado le cobraría hoy",
         "Lo que justifican las características de su crédito",
         "Diferencia"],
    "Tasa": [f"{ea_mercado:.1f} %", f"{ea_riesgo:.1f} %",
             f"{ea_mercado - ea_riesgo:+.2f} puntos"],
}), hide_index=True, use_container_width=True)

st.caption("La segunda cifra sale de un modelo que **no conoce** el sexo, el "
           "grupo étnico ni el departamento del solicitante. Si las dos cifras "
           "son parecidas, el precio se explica por las características del "
           "crédito y no por quién lo pide.")

# --- Bloque 3: contrafactuales -----------------------------------------------
st.header("Qué podría cambiar su tasa")

escenarios = []
con_red = [p for p in PROD_POR_MOD.get(modalidad, [])
           if re.search(r"con\s+recursos de redescuento|provenientes de redescuento",
                        p, re.I)]
if con_red and not re.search(r"con\s+recursos|provenientes", producto, re.I):
    escenarios.append((
        "Pedir la misma línea **con recursos de redescuento** de Bancóldex o Finagro",
        min(predecir(dict(base, producto_de_cr_dito=con_red[0]),
                     M_MERCADO, COLS_MERCADO), 1.0) * tope))

for lbl, val in [
        ("Conseguir **garantía del FAG**",
         "Garantía del fondo agropecuario de garantías (FAG)"),
        ("Conseguir **garantía del FNG**",
         "Garantía fondo nacional de garantías (FNG)"),
        ("Aportar **garantía propia** (hipoteca o prenda)",
         "Garantia idónea o no idónea")]:
    if val != garantia and val in OPC["tipo_de_garant_a"]:
        escenarios.append((lbl, min(predecir(dict(base, tipo_de_garant_a=val),
                                             M_MERCADO, COLS_MERCADO), 1.0) * tope))

if tasa == "FS" and "IB6" in OPC["tipo_de_tasa"]:
    escenarios.append(("Pedir **tasa variable** (indexada al IBR) en vez de fija",
                       min(predecir(dict(base, tipo_de_tasa="IB6"),
                                    M_MERCADO, COLS_MERCADO), 1.0) * tope))

if deudor == "Deudor nuevo en la entidad":
    escenarios.append(("Pedirlo en un banco **donde ya sea cliente**",
                       min(predecir(dict(base, clase_deudor="Deudor de la entidad"),
                                    M_MERCADO, COLS_MERCADO), 1.0) * tope))

if escenarios:
    escenarios.sort(key=lambda t: t[1])
    st.dataframe(pd.DataFrame({
        "Si usted…": [e[0] for e in escenarios[:5]],
        "Tasa estimada": [f"{e[1]:.1f} %" for e in escenarios[:5]],
        "Cambio": [f"{e[1] - ea_mercado:+.1f} pts" for e in escenarios[:5]],
    }), hide_index=True, use_container_width=True)
else:
    st.write("Ya tiene las condiciones más favorables que el modelo contempla.")

st.subheader("Qué preguntar en el banco")
st.markdown("""
Lleve estas preguntas literales a la ventanilla:

1. *"¿Esta línea tiene recursos de redescuento de Bancóldex o de Finagro?"*
2. *"¿Mi crédito puede respaldarse con garantía del FNG o del FAG?"*
3. *"¿Qué tasa me darían si la tomo indexada al IBR en vez de fija?"*
4. *"¿Cuál es la tasa efectiva anual total, incluyendo seguros y estudio de crédito?"*

En nuestros datos, el redescuento es la palanca de precio más grande: la misma
línea puede pasar del 65 % al 17 % efectivo anual según la fuente de los fondos.

Si su tasa le parece alta, **no renuncie al crédito formal**: consulte otras
entidades y los programas de
[Banca de las Oportunidades](https://www.bancadelasoportunidades.gov.co).
""")

# --- Bloque 4: ranking de entidades ------------------------------------------
st.header("Entidades que cobraron menos a perfiles como el suyo")

lista = ENT_POR_MOD.get(modalidad, [])[:25]
if lista:
    Xe = pd.DataFrame([
        dict(base, nombre_entidad=e,
             nombre_tipo_entidad=TIPO_ENT.get(e, "BC-ESTABLECIMIENTO BANCARIO"))
        for e in lista])
    rs = M_MERCADO.predict(Xe[COLS_MERCADO])
    rank = (pd.DataFrame({"Entidad": lista,
                          "Tasa estimada": np.minimum(rs, 1.0) * tope})
            .sort_values("Tasa estimada").head(10).reset_index(drop=True))
    rank.index = range(1, len(rank) + 1)
    rank["Tasa estimada"] = rank["Tasa estimada"].map(lambda v: f"{v:.1f} %")
    st.dataframe(rank, use_container_width=True)
else:
    st.write("No hay suficientes entidades con datos para esta modalidad.")

st.warning(f"**Esto no es una cotización ni una oferta.** Es lo que estas "
           f"entidades cobraron históricamente a créditos parecidos entre "
           f"{CFG['periodo']}. La tasa que le ofrezcan depende del estudio de "
           f"crédito individual.")

# --- Bloque 5: transparencia --------------------------------------------------
with st.expander("Cómo se calculó esta estimación"):
    st.markdown(f"""
Modelo de *gradient boosting* entrenado con **{CFG['n_creditos']:,} créditos
empresariales** reportados por las entidades vigiladas a la Superintendencia
Financiera ({CFG['periodo']}), publicados en datos.gov.co.

**Las variables que más pesan en la estimación:**
""")
    imp = pd.DataFrame({
        "Variable": [k.replace("_", " ") for k in list(CFG["importancias"])[:8]],
        "Peso": list(CFG["importancias"].values())[:8],
    })
    st.dataframe(imp, hide_index=True, use_container_width=True)
    st.markdown("""
Las variables que **menos** pesan son el sexo, el departamento y el grupo étnico
del solicitante: juntas aportan menos que cualquier característica del crédito.

**Limitaciones.** Solo cubre entidades vigiladas; el crédito informal no aparece.
Las tasas están censuradas por el tope de usura, así que el modelo estima el
precio observado y no el que se cobraría sin la restricción legal.

*Ejercicio académico de la Pontificia Universidad Javeriana. No constituye
asesoría financiera.*
""")

st.divider()
st.caption("**Proyecto de Aplicación 1 — Datos Abiertos** · Técnicas de "
           "Aprendizaje de Máquina · Pontificia Universidad Javeriana · "
           "Nameer Chowdhury y Mariana Rodríguez · Fuente: Superintendencia "
           "Financiera de Colombia, datos.gov.co (`w9zh-vetq`, `pare-7x5i`)")
