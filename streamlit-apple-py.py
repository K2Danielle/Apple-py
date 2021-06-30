import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.api.types import is_numeric_dtype, is_string_dtype, is_datetime64_any_dtype
from IPython.display import display
import altair as alt
import pydeck as pdk

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
from mlxtend.frequent_patterns import association_rules

st.title("Apple-PY")
st.title("Analyse comportement d'achats")
st.image("logo.png", width=150)




df = pd.read_csv('data_applepy.csv')


if st.sidebar.checkbox('Description Dataframe'):
    st.subheader("Description du dataframe")
    st.write(df.describe())

if st.sidebar.checkbox('Dataframe'):
    st.subheader("Head Dataframe")
    st.write(df.head())

st.title("Analyse fréquentation du magasin")
st.subheader("Cartographie des communes des clients ayant un compte")

annee = st.sidebar.selectbox('Choisir une année',(2019,2020))

df_cp = df[(df.Encompte==1)&(df.year==annee)]
  
data=df_cp[["nom_commune_postal","latitude","longitude"]]

st.pydeck_chart(pdk.Deck(
     map_style='mapbox://styles/mapbox/light-v9',
     initial_view_state=pdk.ViewState(
         latitude=50.655499,
         longitude=3.140470,
         zoom=11,
         pitch=50,
         bearing=0
         
     ),
     layers=[
         pdk.Layer(
            'HexagonLayer',
            data=data,
            get_position='[longitude, latitude]',
            radius=200,
            opacity=0.8,
            elevation_scale=4,
            elevation_range=[0, 2000],
            extruded=True,
            line_width_min_pixels=1,
           
          ),
         pdk.Layer(
            'ScatterplotLayer',
            data=data,
            get_position='[longitude, latitude]',
            get_color='[0, 250, 0, 160]',
            get_radius=200,
            pickable=True,
            auto_highlight=True,
            get_fill_color=[0,250, 0, 160],
            tooltip=True
         )
     ] ,tooltip={"text": "{nom_commune_postal}"}
 ))


st.subheader("Périodes de fréquentation du magasin")

#Visualisation de la frequentation du magasin par mois
st.write("Visualisations de la frequentation du magasin par mois / jour / heure")
fig3 = plt.figure(figsize=(15, 8))

df['day'].replace(to_replace=0, value=7, inplace=True)

freq_per_year=df.groupby(['year','month',"day","hour"],as_index = False).agg({'Ticket':'count'})
freq_per_year = freq_per_year[freq_per_year['year'].apply(lambda x : x  == annee )]
##Visualisation de la frequentation du magasin par heure de la semaine
ax1 = fig3.add_subplot(3,1,1)
ax1 = sns.lineplot(data=freq_per_year,x='month', hue='year', y='Ticket', palette = "Greens_r")
ax1 = plt.xlabel("Mois")
ax1 = plt.ylabel("Nombre de tickets")
ax1 = plt.title("Fréquentation par mois / an \n ");



## Distribution de la fréquentation du magasin /jour


ax4 = fig3.add_subplot(3,1,2)
ax4 = sns.lineplot(x='day',y='Ticket',hue='year', data=freq_per_year,palette ="Greens_r")
ax4 = plt.ylabel('Nombre de tickets')
ax4 = plt.xlabel('Jour')
ax4 = plt.title('Fréquentation par jour / an\n' )
fig3.tight_layout(h_pad=4, w_pad=4)



##Visualisation de la frequentation du magasin par heure 

ax3 = fig3.add_subplot(3,1,3)
ax3 = sns.lineplot(x='hour',y='Ticket',hue='year',data=freq_per_year,palette ="Greens_r")
ax3 = plt.ylabel('Nombre de tickets')
ax3 = plt.xlabel('Heure')
ax3 = plt.title('Fréquentation par heure / an\n' )
plt.legend();
st.pyplot(fig3)

st.title("Analyse des produits")

## Visualisation des produits les plus vendus par an
st.subheader("Visualisation des 10 produits les plus vendus par an")

# le Libellé court est un nom simple donné à un produit 
# tandis que le Libellé donne aussi des informations sur la famille du produit
lib = st.selectbox('Choisir un libellé ',("Libelle_court","Libelle"))

## Regroupement des artciles par quantités vendues par an
product_sales_per_year=df[(df["Quantite"]>0)].groupby(['year',lib],as_index = False).agg({'Quantite':'sum'})
product_sales_per_year = product_sales_per_year[product_sales_per_year['year'].apply(lambda x : x  == annee )]
#Ensuite, on les trie par quantités vendues par an et on récupère les 10 produits les plus vendus chaque année
max_prod= product_sales_per_year.sort_values(["Quantite"],ascending=False).head(10)

fig4 = plt.figure(figsize=(15, 8))
prod = sns.barplot(x=lib, y='Quantite',data=max_prod,palette ="Greens_r")
prod.set_title("Articles les plus vendus")
plt.xlabel(lib)
plt.ylabel('Quantité')
plt.xticks(rotation = 'vertical')

plt.xticks(rotation = 'vertical')
plt.legend();
st.pyplot(fig4)

st.subheader("Panier d'achat et règles d'association")
st.write("Pour trouver les produits souvent achetés ensemble, nous avons choisi d'appliquer l'algorithme FPGROWTH de MLextend. Nous pouvons définir différents critères de choix, et les données sont sélectionnées par année et par mois. ")


#annee = st.sidebar.selectbox('Choisir une année',(2019,2020))
dict_mois={'janvier':1,'fevrier':2,'mars':3,'avril':4,'mai':5,'juin':6,'juillet':7,'aout':8,'septembre':9,'octobre':10,'novembre':11,'decembre':12}
mois = st.selectbox('Choisir un mois',('janvier','fevrier','mars','avril','mai','juin','juillet','aout','septembre','octobre','novembre','decembre'))
support = st.selectbox('Choisir un seuil pour le support',(0.01,0.02,0.03,0.04,0.05))


df_month=df[(df["month"]==dict_mois[mois])&(df["year"]==annee)]
df_group=df_month.groupby(['Ticket','day'])['Libelle'].apply(lambda group_series: group_series.unique().tolist()).reset_index()
libelle = df_group['Libelle']
te = TransactionEncoder()
te_ary = te.fit(libelle).transform(libelle)
df_ml = pd.DataFrame(te_ary, columns=te.columns_)
#La première phase de l’algorithme est terminée, nous avons identifié tous les itemsets fréquents.
frequent_itemsets = fpgrowth(df_ml, min_support=support, use_colnames=True)
itemsets=frequent_itemsets[frequent_itemsets.itemsets.apply(lambda x: len(x)>1)]

if (len(itemsets)>0):
    itemsets=itemsets.sort_values(by = 'support', ascending = False)

    #représenation itemsets contenant plusieurs produits
    fig5 = plt.figure(figsize=(15, 8))
    asso_prod = fig5.add_subplot(1,1,1)
    plt.title('Associations les plus fréquentes\n')
    asso_prod=sns.barplot(x=itemsets['support'], y=itemsets['itemsets'],data=itemsets ,orient = 'h',palette ="light:#5A9");
    plt.legend();
    st.pyplot(fig5) 
    # Nous calculons maintenant la confiance de chaque règle d’association qui en découle 
    # et nous ne conservons que celles qui satisfont notre critère de confiance.

    # Modification des métriques
    metric = st.selectbox('Choisir la métrique (confidence ou lift)',('confidence','lift'))
    if metric =='confidence':
        seuil_metric= st.slider( 'Choisir un seuil' , min_value=0.3 , max_value=0.9 , value=0.3 , step=0.1)
    elif metric =='lift':
        seuil_metric= st.slider( 'Choisir un seuil' , min_value=1.0, max_value=30.0,value=1.2 , step=0.2)

    rules_all=association_rules(frequent_itemsets, metric=metric, min_threshold=seuil_metric)
    rules_all=rules_all[(rules_all[metric] >seuil_metric)]
    
    
    st.write(rules_all[['antecedents','consequents','support','confidence','lift']])
    
    
else :
    st.write("Pas de regroupement trouvé !")


#st.write("les règles d’association sont des règles du type A implique B. Et sont effectivement souvent utilisées dans l’analyse de paniers d’achats ou pour des outils de recommandation.")
#st.write("Le support : il représente la fiabilité. Ce critère permet de fixer un seuil en dessous duquel les règles ne sont pas considérées comme fiables. Le support d’une règle F1->F2 correspond à la probabilité P(F1 ∩ F2).") 
#st.write("La confiance : elle représente la précision de la règle et peut être vue comme la probabilité conditionnelle Conf(F1->F2)=P(F2| F1).")
#st.write("le lift : il caractérise l’intérêt de la règle, sa force. On a: lift(F1 ->F2)=P(F2 | F1) / P(F2). Un lift supérieur à 1 indique qu’il existe bien un lien entre les 2 éléments")
