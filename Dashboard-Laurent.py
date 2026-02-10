import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from io import BytesIO
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Dashboard Analytique Avancé", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    ::-webkit-scrollbar {width: 12px; height: 12px;}
    ::-webkit-scrollbar-track {background: linear-gradient(to bottom, #f1f1f1, #e0e0e0); border-radius: 10px;}
    ::-webkit-scrollbar-thumb {background: linear-gradient(to bottom, #888, #555); border-radius: 10px;}
    ::-webkit-scrollbar-thumb:hover {background: linear-gradient(to bottom, #555, #333);}
    .alert-critical {background-color: #ff4444; color: white; padding: 15px; border-radius: 10px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .alert-warning {background-color: #ffaa00; color: white; padding: 15px; border-radius: 10px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .alert-success {background-color: #00C851; color: white; padding: 15px; border-radius: 10px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .big-number {font-size: 48px; font-weight: bold; margin: 10px 0;}
    .insight-box {background: #f8f9fa; padding: 20px; border-left: 5px solid #667eea; border-radius: 5px; margin: 15px 0;}
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if 'offer_price' in df.columns:
            df['offer_price'] = pd.to_numeric(df['offer_price'], errors='coerce')
        if 'stock' in df.columns:
            df['stock'] = pd.to_numeric(df['stock'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Erreur: {e}")
        return None

def calculate_revenue_metrics(df):
    metrics = {}
    if 'offer_price' in df.columns and 'stock' in df.columns:
        df_temp = df.dropna(subset=['offer_price', 'stock'])
        metrics['total_potential_revenue'] = (df_temp['offer_price'] * df_temp['stock']).sum()
        
        if 'offer_in_stock' in df.columns:
            out_of_stock = df[df['offer_in_stock'] == False]
            avg_stock = df_temp['stock'].mean() if len(df_temp) > 0 else 5
            metrics['lost_revenue'] = (out_of_stock['offer_price'] * avg_stock).sum()
            total_possible = metrics['total_potential_revenue'] + metrics['lost_revenue']
            metrics['lost_revenue_pct'] = (metrics['lost_revenue'] / total_possible * 100) if total_possible > 0 else 0
    
    if 'stock' in df.columns:
        low_stock = df[df['stock'] < 10]
        metrics['revenue_at_risk'] = (low_stock['offer_price'] * low_stock['stock']).sum()
        metrics['products_at_risk'] = len(low_stock)
        
        # Nouveaux métriques
        overstock = df[df['stock'] > 100]
        metrics['overstock_value'] = (overstock['offer_price'] * overstock['stock']).sum()
        metrics['overstock_count'] = len(overstock)
        
        metrics['optimal_stock_level'] = df_temp['stock'].quantile(0.75) if len(df_temp) > 0 else 20
        metrics['stock_turnover'] = metrics['total_potential_revenue'] / (df_temp['stock'].sum() + 1)
    
    return metrics

def find_revenue_opportunities(df):
    opportunities = []
    if 'offer_in_stock' in df.columns and 'offer_price' in df.columns:
        out_of_stock = df[df['offer_in_stock'] == False].copy()
        if len(out_of_stock) > 0:
            out_of_stock['potential_revenue'] = out_of_stock['offer_price'] * 10
            opportunities.append(('out_of_stock', out_of_stock.nlargest(20, 'potential_revenue')))
        
        if 'stock' in df.columns:
            low_stock = df[(df['stock'] < 10) & (df['stock'] > 0)].copy()
            if len(low_stock) > 0:
                low_stock['additional_revenue'] = low_stock['offer_price'] * (20 - low_stock['stock'])
                opportunities.append(('low_stock', low_stock.nlargest(20, 'additional_revenue')))
        
        in_stock = df[df['offer_in_stock'] == True].copy()
        if len(in_stock) > 0 and 'stock' in df.columns:
            in_stock['current_revenue'] = in_stock['offer_price'] * in_stock['stock']
            opportunities.append(('top_performers', in_stock.nlargest(20, 'current_revenue')))
    
    return opportunities

def analyse_abc(df):
    if 'offer_price' not in df.columns or 'stock' not in df.columns:
        return None
    
    df_abc = df.copy()
    df_abc['revenue'] = df_abc['offer_price'] * df_abc['stock']
    df_abc = df_abc.dropna(subset=['revenue'])
    df_abc = df_abc.sort_values('revenue', ascending=False)
    df_abc['cumulative_revenue'] = df_abc['revenue'].cumsum()
    total_revenue = df_abc['revenue'].sum()
    df_abc['cumulative_pct'] = (df_abc['cumulative_revenue'] / total_revenue * 100)
    df_abc['ABC_Category'] = 'C'
    df_abc.loc[df_abc['cumulative_pct'] <= 80, 'ABC_Category'] = 'A'
    df_abc.loc[(df_abc['cumulative_pct'] > 80) & (df_abc['cumulative_pct'] <= 95), 'ABC_Category'] = 'B'
    
    return df_abc

def predict_stockouts(df, days_ahead=30):
    if 'stock' not in df.columns or 'date' not in df.columns:
        return None
    
    predictions = []
    for product_id in df['product_id'].unique():
        product_data = df[df['product_id'] == product_id].sort_values('date')
        
        if len(product_data) < 3:
            continue
        
        stock_values = product_data['stock'].values
        if len(stock_values) < 2:
            continue
        
        x = np.arange(len(stock_values))
        if len(x) > 1:
            slope = np.polyfit(x, stock_values, 1)[0]
            current_stock = stock_values[-1]
            
            if slope < 0 and current_stock > 0:
                days_to_stockout = int(-current_stock / slope)
                
                if 0 < days_to_stockout <= days_ahead:
                    predictions.append({
                        'product_id': product_id,
                        'current_stock': current_stock,
                        'daily_decrease': -slope,
                        'days_to_stockout': days_to_stockout,
                        'risk_level': 'High' if days_to_stockout <= 7 else 'Medium' if days_to_stockout <= 14 else 'Low'
                    })
    
    return pd.DataFrame(predictions) if predictions else None

def analyze_price_stock_correlation(df):
    if 'offer_price' in df.columns and 'stock' in df.columns:
        corr = df[['offer_price', 'stock']].corr().iloc[0, 1]
        return corr
    return None

def get_top_revenue_products(df, n=20):
    if all(c in df.columns for c in ['offer_price', 'stock', 'title', 'brand']):
        df_rev = df.copy()
        df_rev['revenue'] = df_rev['offer_price'] * df_rev['stock']
        return df_rev.nlargest(n, 'revenue')[['title', 'brand', 'offer_price', 'stock', 'revenue']]
    return None

def calculate_inventory_health(df):
    health_score = 100
    
    if 'offer_in_stock' in df.columns:
        out_of_stock_pct = (df['offer_in_stock'] == False).sum() / len(df) * 100
        health_score -= out_of_stock_pct * 0.5
    
    if 'stock' in df.columns:
        low_stock_pct = (df['stock'] < 10).sum() / len(df) * 100
        health_score -= low_stock_pct * 0.3
        
        overstock_pct = (df['stock'] > 100).sum() / len(df) * 100
        health_score -= overstock_pct * 0.2
    
    return max(0, min(100, health_score))

st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio("Pages:", [
    "🏠 Vue Stratégique", 
    "💰 Revenus Perdus & Opportunités", 
    "📦 Gestion des Stocks", 
    "💵 Analyse des Prix", 
    "📈 Tendances & Prévisions", 
    "🎯 Recommandations Actionnables",
    "⚡ Analyses Avancées",
    "🔍 Insights par Marque",
    "📋 Explorer les Données"
])

df = load_data('all_data_laurent2_CLEAN.csv')

if df is not None:
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filtres")
    
    df_filtered = df.copy()
    
    if 'brand' in df.columns:
        brands = ['Tous'] + sorted(df['brand'].dropna().unique().tolist())
        selected_brand = st.sidebar.selectbox("Marque", brands)
        if selected_brand != 'Tous':
            df_filtered = df_filtered[df_filtered['brand'] == selected_brand]
    
    if 'offer_price' in df.columns:
        min_p, max_p = float(df['offer_price'].min()), float(df['offer_price'].max())
        price_range = st.sidebar.slider("Prix ($)", min_p, max_p, (min_p, max_p))
        df_filtered = df_filtered[(df_filtered['offer_price'] >= price_range[0]) & 
                                  (df_filtered['offer_price'] <= price_range[1])]
    
    if 'offer_in_stock' in df.columns:
        stock_opts = st.sidebar.multiselect("Statut", ['En stock', 'Rupture'], ['En stock', 'Rupture'])
        if 'En stock' in stock_opts and 'Rupture' not in stock_opts:
            df_filtered = df_filtered[df_filtered['offer_in_stock'] == True]
        elif 'Rupture' in stock_opts and 'En stock' not in stock_opts:
            df_filtered = df_filtered[df_filtered['offer_in_stock'] == False]
    
    revenue_metrics = calculate_revenue_metrics(df_filtered)
    opportunities = find_revenue_opportunities(df_filtered)
    inventory_health = calculate_inventory_health(df_filtered)

    # ==================== PAGE: VUE STRATÉGIQUE ====================
    if page == "🏠 Vue Stratégique":
        st.title("🏠 Dashboard Stratégique - Vue d'Ensemble")
        st.markdown("### Tableau de bord exécutif avec tous les KPIs critiques")
        st.markdown("---")
        
        # KPIs principaux
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("💰 Revenue Potentiel", f"${revenue_metrics.get('total_potential_revenue', 0):,.0f}")
        with col2:
            st.metric("⚠️ Revenue Perdu", f"${revenue_metrics.get('lost_revenue', 0):,.0f}", 
                     delta=f"-{revenue_metrics.get('lost_revenue_pct', 0):.1f}%", delta_color="inverse")
        with col3:
            st.metric("🔴 Revenue à Risque", f"${revenue_metrics.get('revenue_at_risk', 0):,.0f}")
        with col4:
            st.metric("📊 Santé Inventaire", f"{inventory_health:.0f}/100", 
                     delta=f"{'✅' if inventory_health > 80 else '⚠️' if inventory_health > 60 else '🔴'}")
        with col5:
            st.metric("📦 Total Produits", f"{df_filtered['product_id'].nunique():,}")
        
        # Deuxième ligne de KPIs
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            if 'offer_in_stock' in df_filtered.columns:
                avail_rate = (df_filtered['offer_in_stock'].sum()/len(df_filtered)*100)
                st.metric("✅ Taux Disponibilité", f"{avail_rate:.1f}%")
        with col2:
            st.metric("📦 Stock Total", f"{df_filtered['stock'].sum():,.0f}" if 'stock' in df_filtered.columns else "N/A")
        with col3:
            st.metric("💵 Prix Moyen", f"${df_filtered['offer_price'].mean():.2f}" if 'offer_price' in df_filtered.columns else "N/A")
        with col4:
            st.metric("🔄 Rotation Stock", f"{revenue_metrics.get('stock_turnover', 0):.2f}x")
        with col5:
            recovery = revenue_metrics.get('lost_revenue', 0) + revenue_metrics.get('revenue_at_risk', 0)
            st.metric("💡 Potentiel Récup.", f"${recovery:,.0f}")
        
        st.markdown("---")
        
        # Alertes Critiques
        st.subheader("🚨 Alertes Critiques & Actions Immédiates")
        ac1, ac2, ac3 = st.columns(3)
        
        with ac1:
            if 'offer_in_stock' in df_filtered.columns:
                out_count = (df_filtered['offer_in_stock'] == False).sum()
                out_pct = (out_count / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
                lost_rev = revenue_metrics.get('lost_revenue', 0)
                
                if out_pct > 20:
                    st.markdown(f'<div class="alert-critical"><h3>🔴 RUPTURES CRITIQUES</h3><p><strong>{out_count}</strong> produits ({out_pct:.1f}%)</p><p>💸 Perte: <strong>${lost_rev:,.0f}</strong></p><p>🎯 Action: Réapprovisionner immédiatement</p></div>', unsafe_allow_html=True)
                elif out_pct > 10:
                    st.markdown(f'<div class="alert-warning"><h3>⚠️ ATTENTION STOCKS</h3><p><strong>{out_count}</strong> produits ({out_pct:.1f}%)</p><p>💸 Impact: <strong>${lost_rev:,.0f}</strong></p><p>🎯 Action: Planifier réapprovisionnement</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="alert-success"><h3>✅ SITUATION NORMALE</h3><p>{out_count} produits en rupture ({out_pct:.1f}%)</p><p>🎯 Continuer la surveillance</p></div>', unsafe_allow_html=True)
        
        with ac2:
            p_risk = revenue_metrics.get('products_at_risk', 0)
            r_risk = revenue_metrics.get('revenue_at_risk', 0)
            if p_risk > 50:
                st.markdown(f'<div class="alert-critical"><h3>🔴 STOCKS FAIBLES</h3><p><strong>{p_risk}</strong> produits</p><p>💰 Valeur: ${r_risk:,.0f}</p><p>🎯 Action: Renforcer stocks prioritaires</p></div>', unsafe_allow_html=True)
            elif p_risk > 20:
                st.markdown(f'<div class="alert-warning"><h3>⚠️ SURVEILLANCE</h3><p><strong>{p_risk}</strong> produits</p><p>💰 Valeur: ${r_risk:,.0f}</p><p>🎯 Action: Surveiller évolution</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-success"><h3>✅ STOCKS STABLES</h3><p>{p_risk} produits en faible stock</p><p>🎯 Situation maîtrisée</p></div>', unsafe_allow_html=True)
        
        with ac3:
            overstock_val = revenue_metrics.get('overstock_value', 0)
            overstock_cnt = revenue_metrics.get('overstock_count', 0)
            if overstock_cnt > 30:
                st.markdown(f'<div class="alert-warning"><h3>⚠️ SURSTOCK</h3><p><strong>{overstock_cnt}</strong> produits</p><p>💰 Valeur immobilisée: ${overstock_val:,.0f}</p><p>🎯 Action: Optimiser rotations</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-success"><h3>✅ OPTIMISÉ</h3><p>{overstock_cnt} produits en surstock</p><p>💰 ${overstock_val:,.0f}</p><p>🎯 Gestion efficace</p></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Graphiques principaux - Ligne 1
        st.subheader("📊 Analyses Visuelles Principales")
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### 💰 Composition du Revenue")
            rev_data = pd.DataFrame({
                'Catégorie': ['💵 Revenue Actuel', '❌ Revenue Perdu', '⚠️ Revenue à Risque', '📦 Surstock'],
                'Valeur': [
                    revenue_metrics.get('total_potential_revenue', 0), 
                    revenue_metrics.get('lost_revenue', 0),
                    revenue_metrics.get('revenue_at_risk', 0),
                    revenue_metrics.get('overstock_value', 0)
                ]
            })
            fig = px.pie(rev_data, values='Valeur', names='Catégorie', hole=0.5,
                        color_discrete_sequence=['#00CC96', '#EF553B', '#FFA15A', '#AB63FA'])
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            total_rev = rev_data['Valeur'].sum()
            st.markdown(f"""
            <div class="insight-box">
            <strong>💡 Insight:</strong> Sur un potentiel total de <strong>${total_rev:,.0f}</strong>, 
            vous perdez actuellement <strong>{(revenue_metrics.get('lost_revenue', 0)/total_rev*100):.1f}%</strong> 
            en ruptures de stock. Optimisation immédiate possible: <strong>${revenue_metrics.get('lost_revenue', 0) + revenue_metrics.get('revenue_at_risk', 0):,.0f}</strong>
            </div>
            """, unsafe_allow_html=True)
        
        with c2:
            st.markdown("#### 🏆 Top 15 Produits par Revenue")
            top_products = get_top_revenue_products(df_filtered, 15)
            if top_products is not None:
                top_products = top_products.head(15)
                fig = px.bar(top_products, y='title', x='revenue', orientation='h',
                           color='revenue', color_continuous_scale='Viridis',
                           labels={'revenue': 'Revenue ($)', 'title': 'Produit'},
                           hover_data=['brand', 'offer_price', 'stock'])
                fig.update_layout(height=400, showlegend=False, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                top_revenue = top_products['revenue'].sum()
                total_revenue = (df_filtered['offer_price'] * df_filtered['stock']).sum()
                st.markdown(f"""
                <div class="insight-box">
                <strong>💡 Insight:</strong> Ces 15 produits représentent <strong>${top_revenue:,.0f}</strong> 
                ({(top_revenue/total_revenue*100):.1f}% du revenue total). Ce sont vos produits stars à protéger absolument.
                </div>
                """, unsafe_allow_html=True)
        
        # Graphiques principaux - Ligne 2
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### 📊 Distribution des Stocks")
            if 'stock' in df_filtered.columns:
                stock_bins = pd.cut(df_filtered['stock'], 
                                   bins=[0, 10, 25, 50, 100, df_filtered['stock'].max()],
                                   labels=['0-10 (Critique)', '11-25 (Faible)', '26-50 (Normal)', 
                                          '51-100 (Bon)', '100+ (Surstock)'])
                stock_dist = stock_bins.value_counts().reset_index()
                stock_dist.columns = ['Niveau', 'Nombre']
                
                fig = px.bar(stock_dist, x='Niveau', y='Nombre', 
                           color='Nombre', color_continuous_scale='RdYlGn',
                           text='Nombre')
                fig.update_traces(texttemplate='%{text}', textposition='outside')
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                critical = (df_filtered['stock'] <= 10).sum()
                st.markdown(f"""
                <div class="insight-box">
                <strong>⚠️ Alerte:</strong> <strong>{critical}</strong> produits sont en niveau critique (≤10 unités). 
                Prioriser le réapprovisionnement de cette catégorie pour éviter les ruptures.
                </div>
                """, unsafe_allow_html=True)
        
        with c2:
            st.markdown("#### 💵 Distribution des Prix")
            if 'offer_price' in df_filtered.columns:
                fig = px.histogram(df_filtered, x='offer_price', nbins=50,
                                 labels={'offer_price': 'Prix ($)', 'count': 'Nombre de produits'},
                                 color_discrete_sequence=['#636EFA'])
                fig.add_vline(x=df_filtered['offer_price'].median(), line_dash="dash", 
                            line_color="red", annotation_text=f"Médiane: ${df_filtered['offer_price'].median():.2f}")
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                price_range = df_filtered['offer_price'].max() - df_filtered['offer_price'].min()
                st.markdown(f"""
                <div class="insight-box">
                <strong>💡 Insight:</strong> Vos prix varient de <strong>${df_filtered['offer_price'].min():.2f}</strong> 
                à <strong>${df_filtered['offer_price'].max():.2f}</strong> (écart: ${price_range:.2f}). 
                Le prix médian est <strong>${df_filtered['offer_price'].median():.2f}</strong>.
                </div>
                """, unsafe_allow_html=True)
        
        # Graphiques principaux - Ligne 3
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### 🎯 Top 10 Marques par Revenue")
            if all(c in df_filtered.columns for c in ['brand', 'offer_price', 'stock']):
                brand_rev = df_filtered.groupby('brand').agg({
                    'offer_price': 'sum',
                    'stock': 'sum',
                    'product_id': 'count'
                }).reset_index()
                brand_rev['revenue'] = brand_rev['offer_price'] * brand_rev['stock']
                brand_rev = brand_rev.nlargest(10, 'revenue')
                
                fig = px.bar(brand_rev, x='revenue', y='brand', orientation='h',
                           color='product_id', color_continuous_scale='Blues',
                           labels={'revenue': 'Revenue ($)', 'brand': 'Marque', 'product_id': 'Nb Produits'},
                           hover_data=['product_id', 'stock'])
                fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                top_brand = brand_rev.iloc[0]
                st.markdown(f"""
                <div class="insight-box">
                <strong>🏆 Leader:</strong> <strong>{top_brand['brand']}</strong> domine avec 
                <strong>${top_brand['revenue']:,.0f}</strong> de revenue sur <strong>{top_brand['product_id']:.0f}</strong> produits.
                </div>
                """, unsafe_allow_html=True)
        
        with c2:
            st.markdown("#### 📈 Statut des Stocks")
            if 'offer_in_stock' in df_filtered.columns and 'stock' in df_filtered.columns:
                status_data = pd.DataFrame({
                    'Statut': ['✅ En Stock (>10)', '⚠️ Stock Faible (1-10)', '❌ Rupture (0)'],
                    'Nombre': [
                        ((df_filtered['offer_in_stock'] == True) & (df_filtered['stock'] > 10)).sum(),
                        ((df_filtered['stock'] > 0) & (df_filtered['stock'] <= 10)).sum(),
                        (df_filtered['offer_in_stock'] == False).sum()
                    ]
                })
                
                fig = px.bar(status_data, x='Statut', y='Nombre',
                           color='Statut', 
                           color_discrete_map={
                               '✅ En Stock (>10)': '#00CC96',
                               '⚠️ Stock Faible (1-10)': '#FFA15A',
                               '❌ Rupture (0)': '#EF553B'
                           },
                           text='Nombre')
                fig.update_traces(texttemplate='%{text}', textposition='outside')
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                health_pct = (status_data[status_data['Statut'] == '✅ En Stock (>10)']['Nombre'].sum() / 
                             status_data['Nombre'].sum() * 100)
                st.markdown(f"""
                <div class="insight-box">
                <strong>📊 Santé Globale:</strong> <strong>{health_pct:.1f}%</strong> de vos produits 
                sont en stock optimal. {'✅ Excellent!' if health_pct > 80 else '⚠️ À améliorer' if health_pct > 60 else '🔴 Action requise'}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Matrice de corrélation
        st.subheader("🔗 Analyse de Corrélation Prix-Stock")
        if 'offer_price' in df_filtered.columns and 'stock' in df_filtered.columns:
            corr = analyze_price_stock_correlation(df_filtered)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = px.scatter(df_filtered, x='offer_price', y='stock',
                               color='stock', size='stock',
                               color_continuous_scale='Viridis',
                               labels={'offer_price': 'Prix ($)', 'stock': 'Stock'},
                               trendline="ols")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown(f"""
                ### Corrélation
                <div class="big-number">{corr:.3f}</div>
                """, unsafe_allow_html=True)
                
                if corr > 0.3:
                    st.success("✅ Corrélation positive: Les produits chers ont tendance à avoir plus de stock")
                elif corr < -0.3:
                    st.warning("⚠️ Corrélation négative: Les produits chers ont moins de stock")
                else:
                    st.info("ℹ️ Faible corrélation: Prix et stock sont indépendants")
                
                st.markdown(f"""
                <div class="insight-box">
                <strong>💡 Stratégie:</strong> {'Maintenir les stocks élevés sur les produits premium' if corr > 0 else 'Revoir la stratégie de stock selon les prix'}
                </div>
                """, unsafe_allow_html=True)

    # ==================== PAGE: REVENUS PERDUS ====================
    elif page == "💰 Revenus Perdus & Opportunités":
        st.title("💰 Analyse Complète des Revenus Perdus")
        st.markdown("### Identifiez et récupérez les opportunités de revenue")
        st.markdown("---")
        
        # KPIs Revenus Perdus
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("💸 Total Perdu", f"${revenue_metrics.get('lost_revenue', 0):,.0f}")
        with c2:
            if 'offer_in_stock' in df_filtered.columns:
                out = (df_filtered['offer_in_stock'] == False).sum()
                st.metric("❌ Produits Rupture", f"{out:,}")
        with c3:
            st.metric("📉 % Revenue Perdu", f"{revenue_metrics.get('lost_revenue_pct', 0):.1f}%")
        with c4:
            out = (df_filtered['offer_in_stock'] == False).sum() if 'offer_in_stock' in df_filtered.columns else 1
            avg = revenue_metrics.get('lost_revenue', 0) / out if out > 0 else 0
            st.metric("💵 Perte Moyenne/Produit", f"${avg:,.0f}")
        with c5:
            recovery_potential = revenue_metrics.get('lost_revenue', 0) + revenue_metrics.get('revenue_at_risk', 0)
            st.metric("🎯 Potentiel Récupération", f"${recovery_potential:,.0f}")
        
        st.markdown("---")
        
        # Visualisations des pertes
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💸 Impact des Ruptures par Marque")
            if 'offer_in_stock' in df_filtered.columns and 'brand' in df_filtered.columns:
                oos_by_brand = df_filtered[df_filtered['offer_in_stock'] == False].groupby('brand').agg({
                    'product_id': 'count',
                    'offer_price': 'sum'
                }).reset_index()
                oos_by_brand['lost_revenue'] = oos_by_brand['offer_price'] * 10
                oos_by_brand = oos_by_brand.nlargest(15, 'lost_revenue')
                
                fig = px.bar(oos_by_brand, x='lost_revenue', y='brand', orientation='h',
                           color='product_id', color_continuous_scale='Reds',
                           labels={'lost_revenue': 'Revenue Perdu ($)', 'brand': 'Marque', 'product_id': 'Nb Ruptures'},
                           hover_data=['product_id'])
                fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Évolution du Taux de Rupture")
            if 'date' in df_filtered.columns and 'offer_in_stock' in df_filtered.columns:
                if df_filtered['date'].notna().any():
                    df_filtered['date_only'] = df_filtered['date'].dt.date
                    daily_oos = df_filtered.groupby('date_only')['offer_in_stock'].apply(
                        lambda x: (x == False).sum() / len(x) * 100
                    ).reset_index()
                    daily_oos.columns = ['date', 'rupture_rate']
                    
                    fig = px.line(daily_oos, x='date', y='rupture_rate',
                                labels={'rupture_rate': 'Taux de Rupture (%)', 'date': 'Date'},
                                markers=True)
                    fig.add_hline(y=10, line_dash="dash", line_color="orange", 
                                annotation_text="Seuil Acceptable (10%)")
                    fig.add_hline(y=20, line_dash="dash", line_color="red",
                                annotation_text="Seuil Critique (20%)")
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Tableaux détaillés
        st.subheader("🎯 Produits à Réapprovisionner en Priorité")
        
        tab1, tab2, tab3 = st.tabs(["🔴 Ruptures Totales", "⚠️ Stocks Critiques (<5)", "📈 Opportunités Cachées"])
        
        with tab1:
            st.markdown("### Produits en rupture totale - Action immédiate requise")
            if 'offer_in_stock' in df_filtered.columns:
                oos = df_filtered[df_filtered['offer_in_stock'] == False].copy()
                oos['potential_revenue'] = oos['offer_price'] * 10
                oos['priority_score'] = oos['offer_price'] * 100
                top_oos = oos.nlargest(30, 'priority_score')[['title', 'brand', 'offer_price', 'potential_revenue', 'priority_score']]
                top_oos.columns = ['Produit', 'Marque', 'Prix', 'Revenue Potentiel (10 unités)', 'Score Priorité']
                
                st.dataframe(top_oos.style.background_gradient(subset=['Revenue Potentiel (10 unités)'], cmap='Reds'), 
                           use_container_width=True, height=600)
                
                st.markdown(f"""
                <div class="insight-box">
                <strong>💡 Action Recommandée:</strong> Commander immédiatement les <strong>{min(10, len(top_oos))}</strong> premiers produits. 
                Gain estimé: <strong>${top_oos['Revenue Potentiel (10 unités)'].head(10).sum():,.0f}</strong>
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### Stocks critiques - Réapprovisionner avant rupture")
            if 'stock' in df_filtered.columns:
                critical = df_filtered[(df_filtered['stock'] > 0) & (df_filtered['stock'] < 5)].copy()
                critical['risk_revenue'] = critical['offer_price'] * critical['stock']
                critical['days_to_stockout'] = (critical['stock'] * 7).astype(int)
                top_critical = critical.nlargest(30, 'risk_revenue')[['title', 'brand', 'stock', 'offer_price', 'risk_revenue', 'days_to_stockout']]
                top_critical.columns = ['Produit', 'Marque', 'Stock Actuel', 'Prix', 'Revenue en Risque', 'Jours Estimés']
                
                st.dataframe(top_critical.style.background_gradient(subset=['Revenue en Risque'], cmap='Oranges'),
                           use_container_width=True, height=600)
                
                st.markdown(f"""
                <div class="insight-box">
                <strong>⚠️ Urgence:</strong> <strong>{len(critical)}</strong> produits risquent la rupture. 
                Revenue en danger: <strong>${critical['risk_revenue'].sum():,.0f}</strong>
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### Produits sous-stockés malgré forte demande potentielle")
            if 'stock' in df_filtered.columns and 'offer_in_stock' in df_filtered.columns:
                understock = df_filtered[(df_filtered['stock'] > 0) & (df_filtered['stock'] < 20) & 
                                        (df_filtered['offer_price'] > df_filtered['offer_price'].median())].copy()
                understock['additional_revenue'] = understock['offer_price'] * (30 - understock['stock'])
                top_understock = understock.nlargest(30, 'additional_revenue')[['title', 'brand', 'stock', 'offer_price', 'additional_revenue']]
                top_understock.columns = ['Produit', 'Marque', 'Stock Actuel', 'Prix', 'Revenue Additionnel Potentiel']
                
                st.dataframe(top_understock.style.background_gradient(subset=['Revenue Additionnel Potentiel'], cmap='Greens'),
                           use_container_width=True, height=600)
                
                st.markdown(f"""
                <div class="insight-box">
                <strong>💡 Opportunité:</strong> Augmenter les stocks de ces produits premium pourrait générer 
                <strong>${understock['additional_revenue'].sum():,.0f}</strong> de revenue supplémentaire.
                </div>
                """, unsafe_allow_html=True)

    # ==================== PAGE: GESTION DES STOCKS ====================
    elif page == "📦 Gestion des Stocks":
        st.title("📦 Gestion Avancée des Stocks")
        st.markdown("### Optimisez vos niveaux de stock pour maximiser le revenue")
        st.markdown("---")
        
        # KPIs Stocks
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            if 'stock' in df_filtered.columns:
                st.metric("📦 Stock Total", f"{df_filtered['stock'].sum():,.0f}")
        with c2:
            if 'stock' in df_filtered.columns:
                st.metric("📊 Stock Moyen", f"{df_filtered['stock'].mean():,.1f}")
        with c3:
            if 'stock' in df_filtered.columns:
                st.metric("📈 Stock Médian", f"{df_filtered['stock'].median():,.0f}")
        with c4:
            if 'stock' in df_filtered.columns:
                critical = (df_filtered['stock'] < 10).sum()
                critical_pct = (critical / len(df_filtered) * 100)
                st.metric("⚠️ Stocks < 10", f"{critical}", delta=f"{critical_pct:.1f}%")
        with c5:
            overstock = revenue_metrics.get('overstock_count', 0)
            st.metric("📦 Surstock (>100)", f"{overstock}")
        
        st.markdown("---")
        
        # Graphiques stocks
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Distribution Complète des Stocks")
            if 'stock' in df_filtered.columns:
                fig = px.histogram(df_filtered, x='stock', nbins=100,
                                 labels={'stock': 'Niveau de Stock', 'count': 'Nombre de Produits'},
                                 color_discrete_sequence=['#636EFA'])
                fig.add_vline(x=10, line_dash="dash", line_color="red", annotation_text="Seuil Critique")
                fig.add_vline(x=df_filtered['stock'].median(), line_dash="dash", 
                            line_color="green", annotation_text=f"Médiane ({df_filtered['stock'].median():.0f})")
                fig.add_vline(x=100, line_dash="dash", line_color="orange", annotation_text="Surstock")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 Matrice Stock vs Prix")
            if 'stock' in df_filtered.columns and 'offer_price' in df_filtered.columns:
                df_filtered['stock_category'] = pd.cut(df_filtered['stock'],
                                                       bins=[0, 10, 50, 100, df_filtered['stock'].max()],
                                                       labels=['Critique', 'Faible', 'Normal', 'Élevé'])
                df_filtered['price_category'] = pd.cut(df_filtered['offer_price'],
                                                      bins=[0, df_filtered['offer_price'].quantile(0.33),
                                                           df_filtered['offer_price'].quantile(0.66),
                                                           df_filtered['offer_price'].max()],
                                                      labels=['Bas', 'Moyen', 'Élevé'])
                
                matrix = df_filtered.groupby(['price_category', 'stock_category']).size().unstack(fill_value=0)
                
                fig = px.imshow(matrix, 
                              labels=dict(x="Niveau de Stock", y="Catégorie de Prix", color="Nombre de Produits"),
                              x=matrix.columns, y=matrix.index,
                              color_continuous_scale='RdYlGn', text_auto=True)
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Recommandations de stock
        st.subheader("🎯 Recommandations de Réajustement des Stocks")
        
        tab1, tab2, tab3 = st.tabs(["⬆️ À Augmenter", "⬇️ À Réduire", "✅ Optimal"])
        
        with tab1:
            st.markdown("### Produits nécessitant une augmentation de stock")
            if 'stock' in df_filtered.columns:
                to_increase = df_filtered[df_filtered['stock'] < 15].copy()
                to_increase['revenue'] = to_increase['offer_price'] * to_increase['stock']
                to_increase['recommended_stock'] = 30
                to_increase['investment_needed'] = to_increase['offer_price'] * (30 - to_increase['stock'])
                to_increase['expected_revenue'] = to_increase['offer_price'] * 30
                
                top_increase = to_increase.nlargest(25, 'expected_revenue')[
                    ['title', 'brand', 'stock', 'recommended_stock', 'offer_price', 'investment_needed', 'expected_revenue']
                ]
                top_increase.columns = ['Produit', 'Marque', 'Stock Actuel', 'Stock Recommandé', 
                                       'Prix Unitaire', 'Investissement', 'Revenue Attendu']
                
                st.dataframe(top_increase.style.background_gradient(subset=['Revenue Attendu'], cmap='Greens'),
                           use_container_width=True, height=500)
                
                total_investment = top_increase['Investissement'].sum()
                total_revenue = top_increase['Revenue Attendu'].sum()
                st.markdown(f"""
                <div class="insight-box">
                <strong>💰 ROI Estimé:</strong> Investir <strong>${total_investment:,.0f}</strong> 
                pour générer <strong>${total_revenue:,.0f}</strong> de revenue potentiel.
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### Produits en surstock - Optimiser la rotation")
            if 'stock' in df_filtered.columns:
                to_decrease = df_filtered[df_filtered['stock'] > 100].copy()
                to_decrease['revenue'] = to_decrease['offer_price'] * to_decrease['stock']
                to_decrease['recommended_stock'] = 50
                to_decrease['excess_stock'] = to_decrease['stock'] - 50
                to_decrease['capital_freed'] = to_decrease['offer_price'] * to_decrease['excess_stock']
                
                top_decrease = to_decrease.nlargest(25, 'capital_freed')[
                    ['title', 'brand', 'stock', 'recommended_stock', 'excess_stock', 'offer_price', 'capital_freed']
                ]
                top_decrease.columns = ['Produit', 'Marque', 'Stock Actuel', 'Stock Recommandé',
                                       'Excédent', 'Prix Unitaire', 'Capital Libérable']
                
                st.dataframe(top_decrease.style.background_gradient(subset=['Capital Libérable'], cmap='Oranges'),
                           use_container_width=True, height=500)
                
                total_freed = top_decrease['Capital Libérable'].sum()
                st.markdown(f"""
                <div class="insight-box">
                <strong>💡 Optimisation:</strong> Réduire ces stocks pourrait libérer <strong>${total_freed:,.0f}</strong> 
                de capital pour investir ailleurs.
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### Produits avec un niveau de stock optimal")
            if 'stock' in df_filtered.columns:
                optimal = df_filtered[(df_filtered['stock'] >= 15) & (df_filtered['stock'] <= 100)].copy()
                optimal['revenue'] = optimal['offer_price'] * optimal['stock']
                
                top_optimal = optimal.nlargest(25, 'revenue')[
                    ['title', 'brand', 'stock', 'offer_price', 'revenue']
                ]
                top_optimal.columns = ['Produit', 'Marque', 'Stock', 'Prix', 'Revenue']
                
                st.dataframe(top_optimal.style.background_gradient(subset=['Revenue'], cmap='Blues'),
                           use_container_width=True, height=500)
                
                st.markdown(f"""
                <div class="insight-box">
                <strong>✅ Bien géré:</strong> <strong>{len(optimal)}</strong> produits sont à un niveau optimal. 
                Continuez cette gestion pour <strong>${optimal['revenue'].sum():,.0f}</strong> de revenue sécurisé.
                </div>
                """, unsafe_allow_html=True)

    # ==================== PAGE: ANALYSE DES PRIX ====================
    elif page == "💵 Analyse des Prix":
        st.title("💵 Analyse Stratégique des Prix")
        st.markdown("### Optimisez votre stratégie de pricing")
        st.markdown("---")
        
        # KPIs Prix
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            if 'offer_price' in df_filtered.columns:
                st.metric("💵 Prix Min", f"${df_filtered['offer_price'].min():,.2f}")
        with c2:
            if 'offer_price' in df_filtered.columns:
                st.metric("💰 Prix Max", f"${df_filtered['offer_price'].max():,.2f}")
        with c3:
            if 'offer_price' in df_filtered.columns:
                st.metric("📊 Prix Moyen", f"${df_filtered['offer_price'].mean():,.2f}")
        with c4:
            if 'offer_price' in df_filtered.columns:
                st.metric("📈 Prix Médian", f"${df_filtered['offer_price'].median():,.2f}")
        with c5:
            if 'offer_price' in df_filtered.columns:
                price_range = df_filtered['offer_price'].max() - df_filtered['offer_price'].min()
                st.metric("📏 Écart", f"${price_range:,.2f}")
        
        st.markdown("---")
        
        # Analyses de prix
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Distribution des Prix (Détaillée)")
            if 'offer_price' in df_filtered.columns:
                fig = px.box(df_filtered, y='offer_price', points='all',
                           labels={'offer_price': 'Prix ($)'},
                           color_discrete_sequence=['#636EFA'])
                fig.add_hline(y=df_filtered['offer_price'].mean(), line_dash="dash",
                            line_color="red", annotation_text=f"Moyenne: ${df_filtered['offer_price'].mean():.2f}")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 💰 Revenue par Tranche de Prix")
            if all(c in df_filtered.columns for c in ['offer_price', 'stock']):
                df_filtered['price_range'] = pd.cut(df_filtered['offer_price'],
                                                    bins=[0, 50, 100, 200, 500, df_filtered['offer_price'].max()],
                                                    labels=['$0-50', '$50-100', '$100-200', '$200-500', '$500+'])
                price_revenue = df_filtered.groupby('price_range').apply(
                    lambda x: (x['offer_price'] * x['stock']).sum()
                ).reset_index()
                price_revenue.columns = ['Tranche', 'Revenue']
                
                fig = px.bar(price_revenue, x='Tranche', y='Revenue',
                           color='Revenue', color_continuous_scale='Viridis',
                           text='Revenue')
                fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Analyses avancées de prix
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Prix Moyen par Marque (Top 15)")
            if all(c in df_filtered.columns for c in ['brand', 'offer_price', 'stock']):
                brand_price = df_filtered.groupby('brand').agg({
                    'offer_price': 'mean',
                    'product_id': 'count',
                    'stock': 'sum'
                }).reset_index()
                brand_price['total_value'] = brand_price['offer_price'] * brand_price['stock']
                brand_price = brand_price.nlargest(15, 'total_value')
                
                fig = px.bar(brand_price, y='brand', x='offer_price', orientation='h',
                           color='product_id', color_continuous_scale='Blues',
                           labels={'offer_price': 'Prix Moyen ($)', 'brand': 'Marque', 'product_id': 'Nb Produits'},
                           hover_data=['stock', 'total_value'])
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📈 Relation Prix-Stock-Revenue")
            if all(c in df_filtered.columns for c in ['offer_price', 'stock']):
                df_sample = df_filtered.sample(min(500, len(df_filtered)))
                df_sample['revenue'] = df_sample['offer_price'] * df_sample['stock']
                
                fig = px.scatter(df_sample, x='offer_price', y='stock',
                               size='revenue', color='revenue',
                               color_continuous_scale='Viridis',
                               labels={'offer_price': 'Prix ($)', 'stock': 'Stock', 'revenue': 'Revenue'},
                               hover_data=['title', 'brand'] if 'title' in df_sample.columns else None)
                st.plotly_chart(fig, use_container_width=True)

    # ==================== PAGE: TENDANCES ====================
    elif page == "📈 Tendances & Prévisions":
        st.title("📈 Tendances Temporelles & Prévisions")
        st.markdown("### Anticipez les évolutions de votre inventaire")
        st.markdown("---")
        
        if 'date' in df_filtered.columns and df_filtered['date'].notna().any():
            df_filtered['date_only'] = df_filtered['date'].dt.date
            df_filtered['year_month'] = df_filtered['date'].dt.to_period('M').astype(str)
            df_filtered['day_of_week'] = df_filtered['date'].dt.day_name()
            
            # Évolutions temporelles
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Évolution Quotidienne du Nombre de Produits")
                daily_count = df_filtered.groupby('date_only').size().reset_index()
                daily_count.columns = ['Date', 'Nombre']
                
                fig = px.line(daily_count, x='Date', y='Nombre',
                            markers=True, color_discrete_sequence=['#636EFA'])
                fig.add_hline(y=daily_count['Nombre'].mean(), line_dash="dash",
                            line_color="red", annotation_text=f"Moyenne: {daily_count['Nombre'].mean():.0f}")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 💰 Évolution du Revenue Potentiel")
                if all(c in df_filtered.columns for c in ['offer_price', 'stock']):
                    daily_rev = df_filtered.groupby('date_only').apply(
                        lambda x: (x['offer_price'] * x['stock']).sum()
                    ).reset_index()
                    daily_rev.columns = ['Date', 'Revenue']
                    
                    fig = px.area(daily_rev, x='Date', y='Revenue',
                                color_discrete_sequence=['#00CC96'])
                    fig.add_hline(y=daily_rev['Revenue'].mean(), line_dash="dash",
                                line_color="red", annotation_text=f"Moyenne: ${daily_rev['Revenue'].mean():,.0f}")
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Prédictions de rupture
            st.subheader("🔮 Prévisions de Ruptures de Stock")
            predictions = predict_stockouts(df_filtered, days_ahead=30)
            
            if predictions is not None and len(predictions) > 0:
                predictions_enriched = predictions.merge(
                    df_filtered[['product_id', 'title', 'brand', 'offer_price']].drop_duplicates('product_id'),
                    on='product_id', how='left'
                )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    high = (predictions['risk_level'] == 'High').sum()
                    st.metric("🔴 Risque Élevé (≤7j)", f"{high} produits")
                with col2:
                    medium = (predictions['risk_level'] == 'Medium').sum()
                    st.metric("🟠 Risque Moyen (8-14j)", f"{medium} produits")
                with col3:
                    low = (predictions['risk_level'] == 'Low').sum()
                    st.metric("🟡 Risque Faible (15-30j)", f"{low} produits")
                
                # Graphique prédictions
                fig = px.scatter(predictions_enriched, x='days_to_stockout', y='current_stock',
                               size='daily_decrease', color='risk_level',
                               color_discrete_map={'High': '#EF553B', 'Medium': '#FFA15A', 'Low': '#00CC96'},
                               hover_data=['title', 'brand', 'offer_price'],
                               labels={'days_to_stockout': 'Jours avant Rupture', 'current_stock': 'Stock Actuel'})
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### ⚠️ Produits Critiques à Surveiller")
                critical_pred = predictions_enriched[predictions_enriched['risk_level'].isin(['High', 'Medium'])].sort_values('days_to_stockout')
                display_pred = critical_pred[['title', 'brand', 'current_stock', 'days_to_stockout', 'daily_decrease', 'risk_level', 'offer_price']]
                display_pred.columns = ['Produit', 'Marque', 'Stock Actuel', 'Jours avant Rupture', 
                                       'Diminution/Jour', 'Niveau Risque', 'Prix']
                st.dataframe(display_pred, use_container_width=True, height=400)
            else:
                st.success("✅ Aucune rupture prévue dans les 30 prochains jours! Excellent travail de gestion.")
        else:
            st.warning("⚠️ Données temporelles insuffisantes pour l'analyse des tendances")

    # ==================== PAGE: RECOMMANDATIONS ====================
    elif page == "🎯 Recommandations Actionnables":
        st.title("🎯 Plan d'Action Stratégique")
        st.markdown("### Recommandations prioritaires basées sur vos données")
        st.markdown("---")
        
        # Calcul du score d'impact pour chaque recommandation
        reco_scores = []
        
        # Recommandation 1: Réapprovisionnement
        if 'offer_in_stock' in df_filtered.columns:
            out_count = (df_filtered['offer_in_stock'] == False).sum()
            lost_rev = revenue_metrics.get('lost_revenue', 0)
            if out_count > 0:
                reco_scores.append({
                    'priority': 1,
                    'title': '🔴 URGENT: Réapprovisionner les Ruptures',
                    'impact': lost_rev,
                    'effort': 'Moyen',
                    'products': out_count,
                    'description': f"Réapprovisionner immédiatement {out_count} produits en rupture",
                    'roi': 'Très Élevé',
                    'timeline': '1-3 jours'
                })
        
        # Recommandation 2: Stocks faibles
        if 'stock' in df_filtered.columns:
            low_stock = df_filtered[(df_filtered['stock'] > 0) & (df_filtered['stock'] < 10)]
            risk_rev = revenue_metrics.get('revenue_at_risk', 0)
            if len(low_stock) > 0:
                reco_scores.append({
                    'priority': 2,
                    'title': '⚠️ IMPORTANT: Renforcer les Stocks Faibles',
                    'impact': risk_rev,
                    'effort': 'Moyen',
                    'products': len(low_stock),
                    'description': f"Augmenter le stock de {len(low_stock)} produits avant rupture",
                    'roi': 'Élevé',
                    'timeline': '3-7 jours'
                })
        
        # Recommandation 3: Surstock
        overstock_val = revenue_metrics.get('overstock_value', 0)
        overstock_cnt = revenue_metrics.get('overstock_count', 0)
        if overstock_cnt > 0:
            reco_scores.append({
                'priority': 3,
                'title': '📦 Optimiser les Surstocks',
                'impact': overstock_val * 0.2,
                'effort': 'Faible',
                'products': overstock_cnt,
                'description': f"Réduire {overstock_cnt} surstocks pour libérer du capital",
                'roi': 'Moyen',
                'timeline': '2-4 semaines'
            })
        
        # Recommandation 4: Produits premium sous-stockés
        if all(c in df_filtered.columns for c in ['offer_price', 'stock', 'offer_in_stock']):
            premium_low = df_filtered[(df_filtered['offer_price'] > df_filtered['offer_price'].quantile(0.75)) &
                                     (df_filtered['stock'] < 20) &
                                     (df_filtered['offer_in_stock'] == True)]
            if len(premium_low) > 0:
                potential = (premium_low['offer_price'] * (30 - premium_low['stock'])).sum()
                reco_scores.append({
                    'priority': 4,
                    'title': '💎 Renforcer les Produits Premium',
                    'impact': potential,
                    'effort': 'Élevé',
                    'products': len(premium_low),
                    'description': f"Augmenter stock de {len(premium_low)} produits premium",
                    'roi': 'Très Élevé',
                    'timeline': '1-2 semaines'
                })
        
        # Affichage des recommandations
        st.subheader("📋 Plan d'Action Prioritaire")
        
        for i, reco in enumerate(sorted(reco_scores, key=lambda x: x['impact'], reverse=True), 1):
            with st.expander(f"**#{i} - {reco['title']}** | Impact: ${reco['impact']:,.0f}", expanded=i<=3):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("💰 Impact Revenue", f"${reco['impact']:,.0f}")
                with col2:
                    st.metric("📦 Produits Affectés", f"{reco['products']}")
                with col3:
                    st.metric("🎯 ROI Attendu", reco['roi'])
                with col4:
                    st.metric("⏱️ Délai", reco['timeline'])
                
                st.markdown(f"""
                <div class="insight-box">
                <h4>📝 Description</h4>
                <p>{reco['description']}</p>
                <p><strong>Effort requis:</strong> {reco['effort']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Actions spécifiques selon le type
                if 'Ruptures' in reco['title']:
                    st.markdown("#### 🎯 Actions Concrètes")
                    st.markdown("""
                    1. **Analyser les top 20 produits** en rupture par revenue potentiel
                    2. **Contacter les fournisseurs** pour réapprovisionnement express
                    3. **Prioriser** les produits avec le plus fort potentiel de vente
                    4. **Mettre en place** des alertes automatiques avant rupture
                    """)
                
                elif 'Stocks Faibles' in reco['title']:
                    st.markdown("#### 🎯 Actions Concrètes")
                    st.markdown("""
                    1. **Identifier** les produits à moins de 7 jours de rupture
                    2. **Augmenter les seuils** de réapprovisionnement automatique
                    3. **Négocier** des délais de livraison plus courts
                    4. **Établir** un niveau de stock de sécurité par catégorie
                    """)
                
                elif 'Surstock' in reco['title']:
                    st.markdown("#### 🎯 Actions Concrètes")
                    st.markdown("""
                    1. **Analyser** les produits immobilisés depuis >90 jours
                    2. **Lancer** des promotions sur les surstocks
                    3. **Réduire** les commandes futures sur ces produits
                    4. **Réallouer** le capital vers des produits à forte rotation
                    """)
                
                elif 'Premium' in reco['title']:
                    st.markdown("#### 🎯 Actions Concrètes")
                    st.markdown("""
                    1. **Augmenter** les stocks des produits premium performants
                    2. **Analyser** la demande historique pour ajuster les niveaux
                    3. **Sécuriser** l'approvisionnement avec des contrats cadres
                    4. **Monitorer** quotidiennement ces produits stratégiques
                    """)
        
        st.markdown("---")
        
        # Résumé financier
        st.subheader("💰 Résumé Financier de l'Optimisation")
        
        total_impact = sum(r['impact'] for r in reco_scores)
        total_products = sum(r['products'] for r in reco_scores)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
            <h3>💰 Gain Potentiel Total</h3>
            <div class="big-number">${total_impact:,.0f}</div>
            <p>En appliquant toutes les recommandations</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
            <h3>📦 Produits à Optimiser</h3>
            <div class="big-number">{total_products}</div>
            <p>Nécessitant une action prioritaire</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            current_health = inventory_health
            potential_health = min(100, current_health + 25)
            st.markdown(f"""
            <div class="metric-card">
            <h3>📈 Amélioration Santé</h3>
            <div class="big-number">{current_health:.0f} → {potential_health:.0f}</div>
            <p>Score de santé inventaire</p>
            </div>
            """, unsafe_allow_html=True)

    # ==================== PAGE: ANALYSES AVANCÉES ====================
    elif page == "⚡ Analyses Avancées":
        st.title("⚡ Analyses Avancées & Intelligence")
        st.markdown("### Insights approfondis et analyses prédictives")
        st.markdown("---")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Analyse ABC", "🎯 Segmentation", "📥 Exports", "🔬 Statistiques"])
        
        with tab1:
            st.subheader("📊 Analyse ABC - Classification Pareto")
            df_abc = analyse_abc(df_filtered)
            
            if df_abc is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    count_a = (df_abc['ABC_Category'] == 'A').sum()
                    revenue_a = df_abc[df_abc['ABC_Category'] == 'A']['revenue'].sum()
                    pct_a = (revenue_a / df_abc['revenue'].sum() * 100)
                    st.metric("🅰️ Catégorie A", f"{count_a} produits ({count_a/len(df_abc)*100:.1f}%)", 
                             f"${revenue_a:,.0f} ({pct_a:.1f}%)")
                with col2:
                    count_b = (df_abc['ABC_Category'] == 'B').sum()
                    revenue_b = df_abc[df_abc['ABC_Category'] == 'B']['revenue'].sum()
                    pct_b = (revenue_b / df_abc['revenue'].sum() * 100)
                    st.metric("🅱️ Catégorie B", f"{count_b} produits ({count_b/len(df_abc)*100:.1f}%)",
                             f"${revenue_b:,.0f} ({pct_b:.1f}%)")
                with col3:
                    count_c = (df_abc['ABC_Category'] == 'C').sum()
                    revenue_c = df_abc[df_abc['ABC_Category'] == 'C']['revenue'].sum()
                    pct_c = (revenue_c / df_abc['revenue'].sum() * 100)
                    st.metric("©️ Catégorie C", f"{count_c} produits ({count_c/len(df_abc)*100:.1f}%)",
                             f"${revenue_c:,.0f} ({pct_c:.1f}%)")
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                with col1:
                    abc_counts = df_abc['ABC_Category'].value_counts()
                    fig = px.pie(values=abc_counts.values, names=abc_counts.index,
                                title="Distribution des Produits par Catégorie",
                                color_discrete_map={'A': '#00CC96', 'B': '#FFA15A', 'C': '#EF553B'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=list(range(len(df_abc))), y=df_abc['cumulative_pct'],
                                            mode='lines', line=dict(color='#636EFA', width=3),
                                            name='Cumul Revenue'))
                    fig.add_hline(y=80, line_dash="dash", line_color="red", 
                                annotation_text="80% - Catégorie A")
                    fig.add_hline(y=95, line_dash="dash", line_color="orange",
                                annotation_text="95% - Catégorie B")
                    fig.update_layout(title="Courbe de Pareto", 
                                    xaxis_title="Produits (classés par revenue décroissant)",
                                    yaxis_title="% Revenue Cumulé")
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### 🏆 Top 30 Produits Catégorie A (High Impact)")
                top_a = df_abc[df_abc['ABC_Category'] == 'A'][['title', 'brand', 'offer_price', 'stock', 'revenue', 'cumulative_pct']].head(30)
                top_a.columns = ['Produit', 'Marque', 'Prix', 'Stock', 'Revenue', '% Cumulé']
                st.dataframe(top_a.style.background_gradient(subset=['Revenue'], cmap='Greens'),
                           use_container_width=True, height=500)
                
                st.markdown(f"""
                <div class="insight-box">
                <strong>💡 Stratégie ABC:</strong><br>
                • <strong>Catégorie A ({count_a} produits)</strong>: Surveillance quotidienne, jamais en rupture<br>
                • <strong>Catégorie B ({count_b} produits)</strong>: Surveillance hebdomadaire, stock de sécurité<br>
                • <strong>Catégorie C ({count_c} produits)</strong>: Revue mensuelle, stock minimal
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            st.subheader("🎯 Segmentation Avancée des Produits")
            
            if all(c in df_filtered.columns for c in ['offer_price', 'stock']):
                # Segmentation en 4 quadrants
                median_price = df_filtered['offer_price'].median()
                median_stock = df_filtered['stock'].median()
                
                df_filtered['segment'] = 'Other'
                df_filtered.loc[(df_filtered['offer_price'] >= median_price) & 
                               (df_filtered['stock'] >= median_stock), 'segment'] = '💎 High Value - High Stock'
                df_filtered.loc[(df_filtered['offer_price'] >= median_price) & 
                               (df_filtered['stock'] < median_stock), 'segment'] = '⚠️ High Value - Low Stock'
                df_filtered.loc[(df_filtered['offer_price'] < median_price) & 
                               (df_filtered['stock'] >= median_stock), 'segment'] = '📦 Low Value - High Stock'
                df_filtered.loc[(df_filtered['offer_price'] < median_price) & 
                               (df_filtered['stock'] < median_stock), 'segment'] = '🔻 Low Value - Low Stock'
                
                # Visualisation
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = px.scatter(df_filtered, x='offer_price', y='stock',
                                   color='segment', size='stock',
                                   color_discrete_map={
                                       '💎 High Value - High Stock': '#00CC96',
                                       '⚠️ High Value - Low Stock': '#EF553B',
                                       '📦 Low Value - High Stock': '#FFA15A',
                                       '🔻 Low Value - Low Stock': '#AB63FA'
                                   },
                                   labels={'offer_price': 'Prix ($)', 'stock': 'Stock'},
                                   hover_data=['title', 'brand'] if 'title' in df_filtered.columns else None)
                    fig.add_vline(x=median_price, line_dash="dash", line_color="gray")
                    fig.add_hline(y=median_stock, line_dash="dash", line_color="gray")
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    segment_counts = df_filtered['segment'].value_counts()
                    st.markdown("### Distribution")
                    for seg, count in segment_counts.items():
                        pct = (count / len(df_filtered) * 100)
                        st.markdown(f"**{seg}**")
                        st.progress(pct/100)
                        st.markdown(f"{count} produits ({pct:.1f}%)")
                        st.markdown("---")
                
                # Recommandations par segment
                st.markdown("### 📋 Actions par Segment")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    <div class="insight-box">
                    <h4>💎 High Value - High Stock</h4>
                    <p><strong>Action:</strong> Excellent! Maintenir ce niveau</p>
                    <ul>
                    <li>Surveiller quotidiennement</li>
                    <li>Assurer la rotation</li>
                    <li>Marketing agressif</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="insight-box">
                    <h4>📦 Low Value - High Stock</h4>
                    <p><strong>Action:</strong> Réduire progressivement</p>
                    <ul>
                    <li>Promotions pour écouler</li>
                    <li>Réduire commandes futures</li>
                    <li>Libérer du capital</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="alert-critical">
                    <h4>⚠️ High Value - Low Stock</h4>
                    <p><strong>Action:</strong> URGENT - Augmenter immédiatement</p>
                    <ul>
                    <li>Réapprovisionner en priorité</li>
                    <li>Augmenter les seuils de commande</li>
                    <li>Éviter les ruptures à tout prix</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="insight-box">
                    <h4>🔻 Low Value - Low Stock</h4>
                    <p><strong>Action:</strong> Évaluer la pertinence</p>
                    <ul>
                    <li>Analyser la rentabilité</li>
                    <li>Considérer l'arrêt si non rentable</li>
                    <li>Ou promouvoir si potentiel</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab3:
            st.subheader("📥 Exporter vos Rapports")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Rapport Excel Complet")
                st.markdown("Inclut: KPIs, données filtrées, analyses ABC, recommandations")
                
                if st.button("📊 Générer Rapport Excel", type="primary"):
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        # Sheet 1: KPIs
                        kpi_data = pd.DataFrame({
                            'Métrique': [
                                'Total Produits',
                                'Revenue Potentiel',
                                'Revenue Perdu',
                                '% Revenue Perdu',
                                'Revenue à Risque',
                                'Produits en Rupture',
                                'Produits Stock Faible',
                                'Score Santé Inventaire',
                                'Stock Total',
                                'Prix Moyen',
                                'Rotation Stock'
                            ],
                            'Valeur': [
                                df_filtered['product_id'].nunique(),
                                f"${revenue_metrics.get('total_potential_revenue', 0):,.0f}",
                                f"${revenue_metrics.get('lost_revenue', 0):,.0f}",
                                f"{revenue_metrics.get('lost_revenue_pct', 0):.1f}%",
                                f"${revenue_metrics.get('revenue_at_risk', 0):,.0f}",
                                (df_filtered['offer_in_stock'] == False).sum() if 'offer_in_stock' in df_filtered.columns else 0,
                                revenue_metrics.get('products_at_risk', 0),
                                f"{inventory_health:.0f}/100",
                                f"{df_filtered['stock'].sum():,.0f}" if 'stock' in df_filtered.columns else 'N/A',
                                f"${df_filtered['offer_price'].mean():,.2f}" if 'offer_price' in df_filtered.columns else 'N/A',
                                f"{revenue_metrics.get('stock_turnover', 0):.2f}x"
                            ]
                        })
                        kpi_data.to_excel(writer, sheet_name='KPIs', index=False)
                        
                        # Sheet 2: Données filtrées
                        df_filtered.to_excel(writer, sheet_name='Données', index=False)
                        
                        # Sheet 3: Analyse ABC
                        if df_abc is not None:
                            df_abc[['product_id', 'title', 'brand', 'offer_price', 'stock', 
                                   'revenue', 'cumulative_pct', 'ABC_Category']].to_excel(
                                writer, sheet_name='Analyse ABC', index=False)
                        
                        # Sheet 4: Ruptures
                        if 'offer_in_stock' in df_filtered.columns:
                            ruptures = df_filtered[df_filtered['offer_in_stock'] == False].copy()
                            ruptures['potential_revenue'] = ruptures['offer_price'] * 10
                            ruptures[['title', 'brand', 'offer_price', 'potential_revenue']].to_excel(
                                writer, sheet_name='Ruptures', index=False)
                    
                    buffer.seek(0)
                    st.download_button(
                        "⬇️ Télécharger le Rapport Excel",
                        buffer,
                        f"rapport_complet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary"
                    )
                    st.success("✅ Rapport généré avec succès!")
            
            with col2:
                st.markdown("### 📄 Export CSV")
                st.markdown("Données filtrées au format CSV pour analyse externe")
                
                cols_to_export = st.multiselect(
                    "Colonnes à exporter",
                    df_filtered.columns.tolist(),
                    default=[c for c in ['product_id', 'title', 'brand', 'offer_price', 'stock', 'offer_in_stock'] 
                            if c in df_filtered.columns]
                )
                
                if cols_to_export and st.button("📄 Générer CSV"):
                    csv = df_filtered[cols_to_export].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Télécharger CSV",
                        csv,
                        f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
                    st.success("✅ CSV généré!")
        
        with tab4:
            st.subheader("🔬 Statistiques Détaillées")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Statistiques des Prix")
                if 'offer_price' in df_filtered.columns:
                    price_stats = df_filtered['offer_price'].describe()
                    stats_df = pd.DataFrame({
                        'Statistique': ['Nombre', 'Moyenne', 'Écart-type', 'Min', '25%', 'Médiane (50%)', '75%', 'Max'],
                        'Valeur': [
                            f"{price_stats['count']:.0f}",
                            f"${price_stats['mean']:.2f}",
                            f"${price_stats['std']:.2f}",
                            f"${price_stats['min']:.2f}",
                            f"${price_stats['25%']:.2f}",
                            f"${price_stats['50%']:.2f}",
                            f"${price_stats['75%']:.2f}",
                            f"${price_stats['max']:.2f}"
                        ]
                    })
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("#### 📦 Statistiques des Stocks")
                if 'stock' in df_filtered.columns:
                    stock_stats = df_filtered['stock'].describe()
                    stats_df = pd.DataFrame({
                        'Statistique': ['Nombre', 'Moyenne', 'Écart-type', 'Min', '25%', 'Médiane (50%)', '75%', 'Max'],
                        'Valeur': [
                            f"{stock_stats['count']:.0f}",
                            f"{stock_stats['mean']:.1f}",
                            f"{stock_stats['std']:.1f}",
                            f"{stock_stats['min']:.0f}",
                            f"{stock_stats['25%']:.0f}",
                            f"{stock_stats['50%']:.0f}",
                            f"{stock_stats['75%']:.0f}",
                            f"{stock_stats['max']:.0f}"
                        ]
                    })
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # ==================== PAGE: INSIGHTS PAR MARQUE ====================
    elif page == "🔍 Insights par Marque":
        st.title("🔍 Analyse Détaillée par Marque")
        st.markdown("### Performance et opportunités par marque")
        st.markdown("---")
        
        if 'brand' in df_filtered.columns:
            # Vue d'ensemble des marques
            brand_summary = df_filtered.groupby('brand').agg({
                'product_id': 'count',
                'offer_price': 'mean',
                'stock': ['sum', 'mean']
            }).reset_index()
            brand_summary.columns = ['brand', 'nb_products', 'avg_price', 'total_stock', 'avg_stock']
            brand_summary['total_value'] = brand_summary['avg_price'] * brand_summary['total_stock']
            brand_summary = brand_summary.sort_values('total_value', ascending=False)
            
            # Top marques
            st.subheader("🏆 Top 15 Marques par Valeur")
            top_brands = brand_summary.head(15)
            
            fig = px.bar(top_brands, x='total_value', y='brand', orientation='h',
                        color='nb_products', color_continuous_scale='Viridis',
                        labels={'total_value': 'Valeur Totale ($)', 'brand': 'Marque', 'nb_products': 'Nb Produits'},
                        hover_data=['avg_price', 'total_stock'])
            fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Sélection d'une marque pour analyse détaillée
            st.subheader("🔎 Analyse Détaillée d'une Marque")
            selected_brand = st.selectbox("Sélectionner une marque", sorted(df_filtered['brand'].unique()))
            
            if selected_brand:
                brand_data = df_filtered[df_filtered['brand'] == selected_brand]
                
                # KPIs de la marque
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("📦 Produits", f"{len(brand_data)}")
                with col2:
                    if 'stock' in brand_data.columns:
                        st.metric("📊 Stock Total", f"{brand_data['stock'].sum():,.0f}")
                with col3:
                    if 'offer_price' in brand_data.columns:
                        st.metric("💵 Prix Moyen", f"${brand_data['offer_price'].mean():.2f}")
                with col4:
                    if all(c in brand_data.columns for c in ['offer_price', 'stock']):
                        brand_revenue = (brand_data['offer_price'] * brand_data['stock']).sum()
                        st.metric("💰 Valeur", f"${brand_revenue:,.0f}")
                with col5:
                    if 'offer_in_stock' in brand_data.columns:
                        availability = (brand_data['offer_in_stock'].sum() / len(brand_data) * 100)
                        st.metric("✅ Disponibilité", f"{availability:.1f}%")
                
                st.markdown("---")
                
                # Graphiques spécifiques à la marque
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📊 Distribution des Prix")
                    if 'offer_price' in brand_data.columns:
                        fig = px.histogram(brand_data, x='offer_price', nbins=30,
                                         labels={'offer_price': 'Prix ($)', 'count': 'Nombre'},
                                         color_discrete_sequence=['#636EFA'])
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### 📦 Distribution des Stocks")
                    if 'stock' in brand_data.columns:
                        fig = px.histogram(brand_data, x='stock', nbins=30,
                                         labels={'stock': 'Stock', 'count': 'Nombre'},
                                         color_discrete_sequence=['#00CC96'])
                        st.plotly_chart(fig, use_container_width=True)
                
                # Top produits de la marque
                st.markdown(f"#### 🏆 Top 20 Produits - {selected_brand}")
                if all(c in brand_data.columns for c in ['offer_price', 'stock', 'title']):
                    brand_data_copy = brand_data.copy()
                    brand_data_copy['revenue'] = brand_data_copy['offer_price'] * brand_data_copy['stock']
                    top_products_brand = brand_data_copy.nlargest(20, 'revenue')[
                        ['title', 'offer_price', 'stock', 'revenue']
                    ]
                    top_products_brand.columns = ['Produit', 'Prix', 'Stock', 'Revenue']
                    st.dataframe(top_products_brand.style.background_gradient(subset=['Revenue'], cmap='Greens'),
                               use_container_width=True, height=400)

    # ==================== PAGE: DONNÉES ====================
    elif page == "📋 Explorer les Données":
        st.title("📋 Explorateur de Données")
        st.markdown("### Vue détaillée et personnalisée de vos données")
        st.markdown("---")
        
        # Sélection des colonnes
        cols = df_filtered.columns.tolist()
        defaults = [c for c in ['product_id', 'title', 'brand', 'offer_price', 'stock', 'offer_in_stock'] if c in cols]
        selected_cols = st.multiselect("Sélectionner les colonnes à afficher", cols, default=defaults)
        
        if selected_cols:
            # Options d'affichage
            col1, col2, col3 = st.columns(3)
            with col1:
                sort_col = st.selectbox("Trier par", selected_cols)
            with col2:
                sort_order = st.radio("Ordre", ["Décroissant", "Croissant"], horizontal=True)
            with col3:
                rows_to_show = st.number_input("Lignes à afficher", min_value=10, max_value=1000, value=100, step=10)
            
            # Tri et affichage
            df_display = df_filtered[selected_cols].copy()
            df_display = df_display.sort_values(sort_col, ascending=(sort_order == "Croissant"))
            
            st.markdown(f"### Affichage de {min(rows_to_show, len(df_display))} lignes sur {len(df_display)} total")
            st.dataframe(df_display.head(rows_to_show), use_container_width=True, height=600)
            
            # Export
            col1, col2 = st.columns(2)
            with col1:
                csv = df_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Télécharger en CSV",
                    csv,
                    f"donnees_filtrees_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    type="primary"
                )
            
            with col2:
                st.metric("📊 Total Lignes", f"{len(df_display):,}")

else:
    st.error("❌ Impossible de charger le fichier 'all_data_laurent2_CLEAN.csv'")
    st.info("Assurez-vous que le fichier est dans le même répertoire que ce script.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"📅 Dernière mise à jour: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
st.sidebar.markdown("🤖 Dashboard Analytique v2.0")
st.sidebar.info("💡 Utilisez les filtres pour affiner vos analyses")