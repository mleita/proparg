import streamlit as st
import pandas as pd
import pickle
import category_encoders as ce
from geopy.geocoders import Nominatim
import folium
import os
import pickle
import rarfile 


# Verificar si el archivo .pkl se extrajo correctamente
pkl_file_path = 'modelo_entrenado.pkl'  # Nombre del archivo .pkl dentro del RAR

# Cargar el archivo CSV
@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('datos_locprov.csv')
    return df

df = load_data()

# Cargar el modelo entrenado
@st.cache(allow_output_mutation=True)
def load_model():
    # Carga tu modelo entrenado desde la ubicación correcta
    with open(pkl_file_path, 'rb') as archivo:
        model = pickle.load(archivo)
    return model

model = load_model()


# Cargar el archivo CSV
@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('datos_locprov.csv')
    return df

df = load_data()

# Cargar el modelo entrenado
@st.cache(allow_output_mutation=True)
def load_model():
    # Carga tu modelo entrenado (asegúrate de que esté en el mismo directorio o proporciona la ruta correcta)
    with open('modelo_entrenado.pkl', 'rb') as archivo:
        model = pickle.load(archivo)
    return model

model = load_model()

# Crear la aplicación Streamlit
st.title('Estimación de precios con un modelo de aprendizaje automático')

# Encabezado desplegable para "¿Qué es esta aplicación?"
with st.expander('**¿Que es esta aplicación?**',expanded=True):
    st.write("Esta aplicación es una muestra del uso de **Machine Learning** en el sector inmobiliario. "
             "Su propósito es proporcionar estimaciones de precios de propiedades en base a diversas características. "
             "Quiero enfatizar que esta aplicación se crea únicamente con **fines académicos** y de demostración del uso del ML en el mercado inmobiliario.")

# Encabezado desplegable para "El Modelo Utilizado"
with st.expander('**El Modelo Utilizado**',expanded=False):
    st.write("Se emplea un modelo de **Machine Learning** llamado **Random Forest Regresor** para realizar estas estimaciones."
             " Un Random Forest es como un grupo de expertos en bienes raíces que trabajan juntos para estimar precios de propiedades. "
             "Cada 'experto' es un árbol de decisión que toma en cuenta diferentes características de las propiedades, "
             "como la ubicación, el número de habitaciones y baños, el tipo de propiedad, entre otros. El poder del Random Forest radica en su capacidad para combinar las opiniones de estos expertos de una manera inteligente."
             " Esto generalmente produce estimaciones más precisas y robustas.")

# Encabezado desplegable para "Origen de los Datos"
with st.expander('**Origen de los Datos**',expanded=False):
    st.write("Los datos utilizados para entrenar este modelo provienen de la base de datos de **Properati** del año 2020. "
             "Esta base de datos contiene una amplia gama de información sobre propiedades, que incluye características como la ubicación, el tipo de propiedad, la cantidad de habitaciones y baños, entre otros.")

# Encabezado desplegable para "Métricas del Modelo"
with st.expander('**Métricas del Modelo**',expanded=False):
    st.write("El modelo ha sido evaluado y tiene un puntaje r2 de **85** en términos de precisión. El **Error Absoluto Medio (MAE)**, "
             "una medida de la precisión de las estimaciones, es de **19000**. Estas métricas nos proporcionan una idea general de la calidad del modelo, "
             "pero nuevamente, es importante recordar que los resultados pueden variar en la vida real.")




# Agregar controles de entrada para las características necesarias, incluyendo las categóricas
rooms = st.slider('Cantidad de habitaciones', min_value=0, max_value=25)
bathrooms = st.slider('Cantidad de baños', min_value=0, max_value=10)
surface_total = st.text_input("Ingrese la superficie total (m2): ")
surface_covered = st.text_input("Ingrese la superficie cubierta (m2): ")
# Agrega tantas características como sean necesarias para tu modelo

# Agregar controles de entrada para las categorías "l2" y "l3"
l2 = st.selectbox('Provinicia:', df['l2'].unique())
filtered_df_l2 = df[df['l2'] == l2]
l3 = st.selectbox('Localidad:', filtered_df_l2['l3'].unique())

with st.expander('No encuentro mi Localidad',expanded=False):
    st.write("Si no encuentras tu localidad dentro del desplegable es porque el modelo no tiene datos de dicha localidad como referencia para realizar la estimación")


# Agregar control de entrada para el tipo de propiedad
property_type = st.selectbox('Tipo de Propiedad', ['Casa', 'Departamento', 'PH', 'Otro', 'Oficina', 'Casa de campo',
       'Local comercial', 'Depósito', 'Lote', 'Cochera'])



# Configura el geocodificador de Nominatim
geolocator = Nominatim(user_agent="geoapiExercises")

calle = st.text_input("Ingrese su calle:")
altura = st.text_input("Ingrese la altura:")

if st.button('Realizar Estimación'):

    # Ingresa la dirección que deseas geocodificar
    direccion = f'{calle},{altura},{l2},{l3}'

    # Utiliza el geocodificador para obtener las coordenadas (latitud y longitud)
    location = geolocator.geocode(direccion)

    if location:
        lat = location.latitude
        lon = location.longitude
    else:
        print('La dirección no pudo ser geocodificada.')
    def load_target_encoder():
        target_encoder = ce.TargetEncoder(cols=['l2', 'l3', 'property_type'])
        return target_encoder

    user_input = pd.DataFrame({
        "l2": [l2],  # Sin codificar, ya que será codificado
        "l3": [l3],  # Sin codificar, ya que será codificado
        "lat": [float(lat)],
        "lon": [float(lon)],
        "property_type": [property_type],  # Sin codificar, ya que será codificado
        "rooms": [float(rooms)],
        "bathrooms": [float(bathrooms)],
        "surface_total": [float(surface_total)],
        "surface_covered": [float(surface_covered)]
    })

    def load_model_encoder():
        # Carga tu modelo entrenado (asegúrate de que esté en el mismo directorio o proporciona la ruta correcta)
        with open('encoder_entrenado.pkl', 'rb') as archivo:
            model_encoder = pickle.load(archivo)
        return model_encoder

    target_encoder = load_model_encoder()
    # Crear el codificador de destino (Target Encoder)

    # Aplicar la codificación de destino al DataFrame de entrada
    user_input_encoded = target_encoder.transform(user_input)



    # Realizar una predicción utilizando el modelo entrenado
    prediction = model.predict(user_input_encoded)
    st.subheader(f'El precio estimado de la propiedad en {calle},{altura}, de la ciudad de {l2}, es de  USD {int(prediction)}')

    # Crear un mapa interactivo en Streamlit
    st.title('Mapa de Ubicación')

    # Crear un objeto de mapa con la ubicación seleccionada
    m = folium.Map(location=[lat, lon], zoom_start=15)

    # Agregar un marcador al mapa en la ubicación seleccionada
    folium.Marker([lat, lon], tooltip=f'Precio estimado: USD {int(prediction)}').add_to(m)

    # Mostrar el mapa en Streamlit
    st.components.v1.html(m._repr_html_(), width=800, height=600)

    
# Definir contenido en la barra lateral (sidebar)
if st.button('Información del autor'):
    st.sidebar.image('maxi.png', width = 150)
    st.sidebar.title('Acerca de Mí')
    st.sidebar.write('¡Hola! Mi nombre es Maximiliano Leita, un entusiasta la programación, la ciencia de datos y profesional del márketing.')

    # Agregar información adicional
    st.sidebar.header('Contactame')
    imagen_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/LinkedIn_logo_initials.png/600px-LinkedIn_logo_initials.png?20140125013055'
    # URL a la que quieres que la imagen enlace
    enlace_url = 'https://www.linkedin.com/in/alexis-maximiliano-leita-886a5976/'

    ancho = '20%'
    # Mostrar la imagen como un enlace
    st.sidebar.markdown(f'<a href="{enlace_url}" target="_blank"><img src="{imagen_url}" alt="Mi Imagen" style="width: {ancho};"></a>', unsafe_allow_html=True)
