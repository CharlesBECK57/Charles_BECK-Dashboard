import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from io import BytesIO
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Advanced Analytics Dashboard", page_icon="📊", layout="wide")

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

        # Remove unwanted columns
        columns_to_drop = ['product_id', 'offer_offer_id']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

        # Recreate product_id as unique identifier (required for dashboard)
        if 'product_id' not in df.columns:
            df['product_id'] = range(len(df))

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if 'offer_price' in df.columns:
            df['offer_price'] = pd.to_numeric(df['offer_price'], errors='coerce')
        if 'stock' in df.columns:
            df['stock'] = pd.to_numeric(df['stock'], errors='coerce')
        if 'offer_in_stock' in df.columns:
            # Convert to boolean (True/False) to avoid type errors
            df['offer_in_stock'] = df['offer_in_stock'].astype(bool)

        # Normalize brand names (merge duplicates)
        if 'brand' in df.columns:
            # Clean whitespace
            df['brand'] = df['brand'].str.strip()
            # Replace all variants
            df['brand'] = df['brand'].str.replace('AUXPORTESDUNATUREL', 'AUX PORTES DU NATUREL', regex=False)
            df['brand'] = df['brand'].replace('AUXPORTESDUNATUREL', 'AUX PORTES DU NATUREL')

        return df
    except Exception as e:
        st.error(f"Error: {e}")
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
        
        # New metrics
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
    "🏠 Strategic Overview",
    "💰 Lost Revenue & Opportunities",
    "📦 Stock Management",
    "💵 Pricing Analysis",
    "🎯 Actionable Recommendations",
    "⚡ Advanced Analytics",
    "🔍 Brand Insights",
    "📋 Data Explorer"
])

st.sidebar.markdown("---")
st.sidebar.header("📂 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=['csv'])

if uploaded_file is not None:
    df = load_data(uploaded_file)
elif os.path.exists('all_data_laurent2.csv'):
    df = load_data('all_data_laurent2.csv')
else:
    df = None

if df is not None:
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filters")

    df_filtered = df.copy()

    if 'brand' in df.columns:
        brands = ['All'] + sorted(df['brand'].dropna().unique().tolist())
        selected_brand = st.sidebar.selectbox("Brand", brands)
        if selected_brand != 'All':
            df_filtered = df_filtered[df_filtered['brand'] == selected_brand]

    if 'offer_price' in df.columns:
        min_p, max_p = float(df['offer_price'].min()), float(df['offer_price'].max())
        price_range = st.sidebar.slider("Price ($)", min_p, max_p, (min_p, max_p))
        df_filtered = df_filtered[(df_filtered['offer_price'] >= price_range[0]) &
                                  (df_filtered['offer_price'] <= price_range[1])]

    if 'offer_in_stock' in df.columns:
        stock_opts = st.sidebar.multiselect("Status", ['In Stock', 'Out of Stock'], ['In Stock', 'Out of Stock'])
        if 'In Stock' in stock_opts and 'Out of Stock' not in stock_opts:
            df_filtered = df_filtered[df_filtered['offer_in_stock'] == True]
        elif 'Out of Stock' in stock_opts and 'In Stock' not in stock_opts:
            df_filtered = df_filtered[df_filtered['offer_in_stock'] == False]
    
    revenue_metrics = calculate_revenue_metrics(df_filtered)
    opportunities = find_revenue_opportunities(df_filtered)
    inventory_health = calculate_inventory_health(df_filtered)

    # ==================== PAGE: STRATEGIC OVERVIEW ====================
    if page == "🏠 Strategic Overview":
        st.title("🏠 Strategic Dashboard - Overview")
        st.markdown("### Executive dashboard with all critical KPIs")
        st.markdown("---")

        # Main KPIs
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("💰 Potential Revenue", f"${revenue_metrics.get('total_potential_revenue', 0):,.0f}")
        with col2:
            st.metric("⚠️ Lost Revenue", f"${revenue_metrics.get('lost_revenue', 0):,.0f}",
                     delta=f"-{revenue_metrics.get('lost_revenue_pct', 0):.1f}%", delta_color="inverse")
        with col3:
            st.metric("🔴 Revenue at Risk", f"${revenue_metrics.get('revenue_at_risk', 0):,.0f}")
        with col4:
            st.metric("📊 Inventory Health", f"{inventory_health:.0f}/100",
                     delta=f"{'✅' if inventory_health > 80 else '⚠️' if inventory_health > 60 else '🔴'}")
        with col5:
            st.metric("📦 Total Products", f"{int(df_filtered['product_id'].nunique()):,}")

        # Second line of KPIs
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            if 'offer_in_stock' in df_filtered.columns:
                avail_rate = ((df_filtered['offer_in_stock'] == True).sum()/len(df_filtered)*100)
                st.metric("✅ Availability Rate", f"{avail_rate:.1f}%")
        with col2:
            st.metric("📦 Total Stock", f"{df_filtered['stock'].sum():,.0f}" if 'stock' in df_filtered.columns else "N/A")
        with col3:
            st.metric("💵 Average Price", f"${df_filtered['offer_price'].mean():.2f}" if 'offer_price' in df_filtered.columns else "N/A")
        with col4:
            st.metric("🔄 Stock Turnover", f"{revenue_metrics.get('stock_turnover', 0):.2f}x")
        with col5:
            recovery = revenue_metrics.get('lost_revenue', 0) + revenue_metrics.get('revenue_at_risk', 0)
            st.metric("💡 Recovery Potential", f"${recovery:,.0f}")
        
        st.markdown("---")

        # Critical Alerts
        st.subheader("🚨 Critical Alerts & Immediate Actions")
        ac1, ac2, ac3 = st.columns(3)

        with ac1:
            if 'offer_in_stock' in df_filtered.columns:
                out_count = (df_filtered['offer_in_stock'] == False).sum()
                out_pct = (out_count / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
                lost_rev = revenue_metrics.get('lost_revenue', 0)

                if out_pct > 20:
                    st.markdown(f'<div class="alert-critical"><h3>🔴 CRITICAL STOCKOUTS</h3><p><strong>{out_count}</strong> products ({out_pct:.1f}%)</p><p>💸 Loss: <strong>${lost_rev:,.0f}</strong></p><p>🎯 Action: Restock immediately</p></div>', unsafe_allow_html=True)
                elif out_pct > 10:
                    st.markdown(f'<div class="alert-warning"><h3>⚠️ STOCK WARNING</h3><p><strong>{out_count}</strong> products ({out_pct:.1f}%)</p><p>💸 Impact: <strong>${lost_rev:,.0f}</strong></p><p>🎯 Action: Plan restocking</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="alert-success"><h3>✅ NORMAL SITUATION</h3><p>{out_count} products out of stock ({out_pct:.1f}%)</p><p>🎯 Continue monitoring</p></div>', unsafe_allow_html=True)

        with ac2:
            p_risk = revenue_metrics.get('products_at_risk', 0)
            r_risk = revenue_metrics.get('revenue_at_risk', 0)
            if p_risk > 50:
                st.markdown(f'<div class="alert-critical"><h3>🔴 LOW STOCKS</h3><p><strong>{p_risk}</strong> products</p><p>💰 Value: ${r_risk:,.0f}</p><p>🎯 Action: Boost priority stocks</p></div>', unsafe_allow_html=True)
            elif p_risk > 20:
                st.markdown(f'<div class="alert-warning"><h3>⚠️ MONITORING</h3><p><strong>{p_risk}</strong> products</p><p>💰 Value: ${r_risk:,.0f}</p><p>🎯 Action: Monitor evolution</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-success"><h3>✅ STABLE STOCKS</h3><p>{p_risk} products with low stock</p><p>🎯 Situation under control</p></div>', unsafe_allow_html=True)

        with ac3:
            overstock_val = revenue_metrics.get('overstock_value', 0)
            overstock_cnt = revenue_metrics.get('overstock_count', 0)
            if overstock_cnt > 30:
                st.markdown(f'<div class="alert-warning"><h3>⚠️ OVERSTOCK</h3><p><strong>{overstock_cnt}</strong> products</p><p>💰 Tied-up value: ${overstock_val:,.0f}</p><p>🎯 Action: Optimize turnover</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-success"><h3>✅ OPTIMIZED</h3><p>{overstock_cnt} overstocked products</p><p>💰 ${overstock_val:,.0f}</p><p>🎯 Efficient management</p></div>', unsafe_allow_html=True)
        
        st.markdown("---")

        # Main Charts - Line 1
        st.subheader("📊 Main Visual Analysis")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 💰 Revenue Composition")
            rev_data = pd.DataFrame({
                'Category': ['💵 Current Revenue', '❌ Lost Revenue', '⚠️ Revenue at Risk', '📦 Overstock'],
                'Value': [
                    revenue_metrics.get('total_potential_revenue', 0),
                    revenue_metrics.get('lost_revenue', 0),
                    revenue_metrics.get('revenue_at_risk', 0),
                    revenue_metrics.get('overstock_value', 0)
                ]
            })
            fig = px.pie(rev_data, values='Value', names='Category', hole=0.5,
                        color_discrete_sequence=['#00CC96', '#EF553B', '#FFA15A', '#AB63FA'])
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

            total_rev = rev_data['Value'].sum()
            st.markdown(f"""
            <div class="insight-box">
            <strong>💡 Insight:</strong> Out of a total potential of <strong>${total_rev:,.0f}</strong>,
            you are currently losing <strong>{(revenue_metrics.get('lost_revenue', 0)/total_rev*100):.1f}%</strong>
            in stockouts. Immediate optimization possible: <strong>${revenue_metrics.get('lost_revenue', 0) + revenue_metrics.get('revenue_at_risk', 0):,.0f}</strong>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("#### 🏆 Top 15 Products by Revenue")
            top_products = get_top_revenue_products(df_filtered, 15)
            if top_products is not None:
                top_products = top_products.head(15)
                fig = px.bar(top_products, y='title', x='revenue', orientation='h',
                           color='revenue', color_continuous_scale='Viridis',
                           labels={'revenue': 'Revenue ($)', 'title': 'Product'},
                           hover_data=['brand', 'offer_price', 'stock'])
                fig.update_layout(height=400, showlegend=False, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

                top_revenue = top_products['revenue'].sum()
                total_revenue = (df_filtered['offer_price'] * df_filtered['stock']).sum()
                st.markdown(f"""
                <div class="insight-box">
                <strong>💡 Insight:</strong> These 15 products represent <strong>${top_revenue:,.0f}</strong>
                ({(top_revenue/total_revenue*100):.1f}% of total revenue). These are your star products to protect absolutely.
                </div>
                """, unsafe_allow_html=True)
        
        # Main Charts - Line 2
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 📊 Stock Distribution")
            if 'stock' in df_filtered.columns:
                stock_bins = pd.cut(df_filtered['stock'],
                                   bins=[0, 10, 25, 50, 100, df_filtered['stock'].max()],
                                   labels=['0-10 (Critical)', '11-25 (Low)', '26-50 (Normal)',
                                          '51-100 (Good)', '100+ (Overstock)'])
                stock_dist = stock_bins.value_counts().reset_index()
                stock_dist.columns = ['Level', 'Count']

                fig = px.bar(stock_dist, x='Level', y='Count',
                           color='Count', color_continuous_scale='RdYlGn',
                           text='Count')
                fig.update_traces(texttemplate='%{text}', textposition='outside')
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                critical = (df_filtered['stock'] <= 10).sum()
                st.markdown(f"""
                <div class="insight-box">
                <strong>⚠️ Alert:</strong> <strong>{critical}</strong> products are at critical level (≤10 units).
                Prioritize restocking this category to avoid stockouts.
                </div>
                """, unsafe_allow_html=True)

        with c2:
            st.markdown("#### 💵 Price Distribution")
            if 'offer_price' in df_filtered.columns:
                fig = px.histogram(df_filtered, x='offer_price', nbins=50,
                                 labels={'offer_price': 'Price ($)', 'count': 'Number of products'},
                                 color_discrete_sequence=['#636EFA'])
                fig.add_vline(x=df_filtered['offer_price'].median(), line_dash="dash",
                            line_color="red", annotation_text=f"Median: ${df_filtered['offer_price'].median():.2f}")
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                price_range = df_filtered['offer_price'].max() - df_filtered['offer_price'].min()
                st.markdown(f"""
                <div class="insight-box">
                <strong>💡 Insight:</strong> Your prices range from <strong>${df_filtered['offer_price'].min():.2f}</strong>
                to <strong>${df_filtered['offer_price'].max():.2f}</strong> (range: ${price_range:.2f}).
                The median price is <strong>${df_filtered['offer_price'].median():.2f}</strong>.
                </div>
                """, unsafe_allow_html=True)
        
        # Main Charts - Line 3
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 🎯 Top 10 Brands by Revenue")
            if all(c in df_filtered.columns for c in ['brand', 'offer_price', 'stock']):
                brand_rev = df_filtered.groupby('brand').agg({
                    'offer_price': 'sum',
                    'stock': 'sum',
                    'product_id': 'count'
                }).reset_index()
                brand_rev.columns = ['brand', 'total_price', 'total_stock', 'nb_products']
                brand_rev['revenue'] = brand_rev['total_price'] * brand_rev['total_stock']
                brand_rev = brand_rev.nlargest(10, 'revenue')

                fig = px.bar(brand_rev, x='revenue', y='brand', orientation='h',
                           color='nb_products', color_continuous_scale='Blues',
                           labels={'revenue': 'Revenue ($)', 'brand': 'Brand', 'nb_products': 'Nb Products'},
                           hover_data=['nb_products', 'total_stock'])
                fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

                top_brand = brand_rev.iloc[0]
                st.markdown(f"""
                <div class="insight-box">
                <strong>🏆 Leader:</strong> <strong>{top_brand['brand']}</strong> dominates with
                <strong>${top_brand['revenue']:,.0f}</strong> in revenue across <strong>{int(top_brand['nb_products'])}</strong> products.
                </div>
                """, unsafe_allow_html=True)

        with c2:
            st.markdown("#### 📈 Stock Status")
            if 'offer_in_stock' in df_filtered.columns and 'stock' in df_filtered.columns:
                status_data = pd.DataFrame({
                    'Status': ['✅ In Stock (>10)', '⚠️ Low Stock (1-10)', '❌ Out of Stock (0)'],
                    'Count': [
                        ((df_filtered['offer_in_stock'] == True) & (df_filtered['stock'] > 10)).sum(),
                        ((df_filtered['stock'] > 0) & (df_filtered['stock'] <= 10)).sum(),
                        (df_filtered['offer_in_stock'] == False).sum()
                    ]
                })

                fig = px.bar(status_data, x='Status', y='Count',
                           color='Status',
                           color_discrete_map={
                               '✅ In Stock (>10)': '#00CC96',
                               '⚠️ Low Stock (1-10)': '#FFA15A',
                               '❌ Out of Stock (0)': '#EF553B'
                           },
                           text='Count')
                fig.update_traces(texttemplate='%{text}', textposition='outside')
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                health_pct = (status_data[status_data['Status'] == '✅ In Stock (>10)']['Count'].sum() /
                             status_data['Count'].sum() * 100)
                st.markdown(f"""
                <div class="insight-box">
                <strong>📊 Overall Health:</strong> <strong>{health_pct:.1f}%</strong> of your products
                are at optimal stock levels. {'✅ Excellent!' if health_pct > 80 else '⚠️ Room for improvement' if health_pct > 60 else '🔴 Action required'}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

    # ==================== PAGE: LOST REVENUE ====================
    elif page == "💰 Lost Revenue & Opportunities":
        st.title("💰 Complete Lost Revenue Analysis")
        st.markdown("### Identify and recover revenue opportunities")
        st.markdown("---")

        # Lost Revenue KPIs
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("💸 Total Lost", f"${revenue_metrics.get('lost_revenue', 0):,.0f}")
        with c2:
            if 'offer_in_stock' in df_filtered.columns:
                out = (df_filtered['offer_in_stock'] == False).sum()
                st.metric("❌ Out of Stock Products", f"{out:,}")
        with c3:
            st.metric("📉 % Revenue Lost", f"{revenue_metrics.get('lost_revenue_pct', 0):.1f}%")
        with c4:
            out = (df_filtered['offer_in_stock'] == False).sum() if 'offer_in_stock' in df_filtered.columns else 1
            avg = revenue_metrics.get('lost_revenue', 0) / out if out > 0 else 0
            st.metric("💵 Avg Loss/Product", f"${avg:,.0f}")
        with c5:
            recovery_potential = revenue_metrics.get('lost_revenue', 0) + revenue_metrics.get('revenue_at_risk', 0)
            st.metric("🎯 Recovery Potential", f"${recovery_potential:,.0f}")

        st.markdown("---")

        # Loss Visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 💸 Stockout Impact by Brand")
            if 'offer_in_stock' in df_filtered.columns and 'brand' in df_filtered.columns:
                oos_by_brand = df_filtered[df_filtered['offer_in_stock'] == False].groupby('brand').agg({
                    'product_id': 'count',
                    'offer_price': 'sum'
                }).reset_index()
                oos_by_brand.columns = ['brand', 'nb_stockouts', 'total_price']
                oos_by_brand['lost_revenue'] = oos_by_brand['total_price'] * 10
                oos_by_brand = oos_by_brand.nlargest(15, 'lost_revenue')

                fig = px.bar(oos_by_brand, x='lost_revenue', y='brand', orientation='h',
                           color='nb_stockouts', color_continuous_scale='Reds',
                           labels={'lost_revenue': 'Lost Revenue ($)', 'brand': 'Brand', 'nb_stockouts': 'Nb Stockouts'},
                           hover_data=['nb_stockouts'])
                fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### 📊 Stockout Rate Evolution")
            if 'date' in df_filtered.columns and 'offer_in_stock' in df_filtered.columns:
                if df_filtered['date'].notna().any():
                    df_filtered['date_only'] = df_filtered['date'].dt.date
                    daily_oos = df_filtered.groupby('date_only')['offer_in_stock'].apply(
                        lambda x: (x == False).sum() / len(x) * 100
                    ).reset_index()
                    daily_oos.columns = ['date', 'stockout_rate']

                    fig = px.line(daily_oos, x='date', y='stockout_rate',
                                labels={'stockout_rate': 'Stockout Rate (%)', 'date': 'Date'},
                                markers=True)
                    fig.add_hline(y=10, line_dash="dash", line_color="orange",
                                annotation_text="Acceptable Threshold (10%)")
                    fig.add_hline(y=20, line_dash="dash", line_color="red",
                                annotation_text="Critical Threshold (20%)")
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

    # ==================== PAGE: STOCK MANAGEMENT ====================
    elif page == "📦 Stock Management":
        st.title("📦 Advanced Stock Management")
        st.markdown("### Optimize your stock levels to maximize revenue")
        st.markdown("---")

        # Stock KPIs
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            if 'stock' in df_filtered.columns:
                st.metric("📦 Total Stock", f"{df_filtered['stock'].sum():,.0f}")
        with c2:
            if 'stock' in df_filtered.columns:
                st.metric("📊 Average Stock", f"{df_filtered['stock'].mean():,.1f}")
        with c3:
            if 'stock' in df_filtered.columns:
                st.metric("📈 Median Stock", f"{df_filtered['stock'].median():,.0f}")
        with c4:
            if 'stock' in df_filtered.columns:
                critical = (df_filtered['stock'] < 10).sum()
                critical_pct = (critical / len(df_filtered) * 100)
                st.metric("⚠️ Stocks < 10", f"{critical}", delta=f"{critical_pct:.1f}%")
        with c5:
            overstock = revenue_metrics.get('overstock_count', 0)
            st.metric("📦 Overstock (>100)", f"{overstock}")

        st.markdown("---")

        # Stock Charts
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📊 Complete Stock Distribution")
            if 'stock' in df_filtered.columns:
                fig = px.histogram(df_filtered, x='stock', nbins=100,
                                 labels={'stock': 'Stock Level', 'count': 'Number of Products'},
                                 color_discrete_sequence=['#636EFA'])
                fig.add_vline(x=10, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
                fig.add_vline(x=df_filtered['stock'].median(), line_dash="dash",
                            line_color="green", annotation_text=f"Median ({df_filtered['stock'].median():.0f})")
                fig.add_vline(x=100, line_dash="dash", line_color="orange", annotation_text="Overstock")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### 🎯 Stock vs Price Matrix")
            if 'stock' in df_filtered.columns and 'offer_price' in df_filtered.columns:
                df_filtered['stock_category'] = pd.cut(df_filtered['stock'],
                                                       bins=[0, 10, 50, 100, df_filtered['stock'].max()],
                                                       labels=['Critical', 'Low', 'Normal', 'High'])
                df_filtered['price_category'] = pd.cut(df_filtered['offer_price'],
                                                      bins=[0, df_filtered['offer_price'].quantile(0.33),
                                                           df_filtered['offer_price'].quantile(0.66),
                                                           df_filtered['offer_price'].max()],
                                                      labels=['Low', 'Medium', 'High'])

                matrix = df_filtered.groupby(['price_category', 'stock_category']).size().unstack(fill_value=0)

                fig = px.imshow(matrix,
                              labels=dict(x="Stock Level", y="Price Category", color="Number of Products"),
                              x=matrix.columns, y=matrix.index,
                              color_continuous_scale='RdYlGn', text_auto=True)
                st.plotly_chart(fig, use_container_width=True)
        

    # ==================== PAGE: PRICING ANALYSIS ====================
    elif page == "💵 Pricing Analysis":
        st.title("💵 Strategic Pricing Analysis")
        st.markdown("### Optimize your pricing strategy")
        st.markdown("---")

        # Price KPIs
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            if 'offer_price' in df_filtered.columns:
                st.metric("💵 Min Price", f"${df_filtered['offer_price'].min():,.2f}")
        with c2:
            if 'offer_price' in df_filtered.columns:
                st.metric("💰 Max Price", f"${df_filtered['offer_price'].max():,.2f}")
        with c3:
            if 'offer_price' in df_filtered.columns:
                st.metric("📊 Average Price", f"${df_filtered['offer_price'].mean():,.2f}")
        with c4:
            if 'offer_price' in df_filtered.columns:
                st.metric("📈 Median Price", f"${df_filtered['offer_price'].median():,.2f}")
        with c5:
            if 'offer_price' in df_filtered.columns:
                price_range = df_filtered['offer_price'].max() - df_filtered['offer_price'].min()
                st.metric("📏 Range", f"${price_range:,.2f}")

        st.markdown("---")

        # Price Analysis
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📊 Price Distribution (Detailed)")
            if 'offer_price' in df_filtered.columns:
                fig = px.box(df_filtered, y='offer_price', points='all',
                           labels={'offer_price': 'Price ($)'},
                           color_discrete_sequence=['#636EFA'])
                fig.add_hline(y=df_filtered['offer_price'].mean(), line_dash="dash",
                            line_color="red", annotation_text=f"Average: ${df_filtered['offer_price'].mean():.2f}")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### 📈 Price-Stock-Revenue Relationship")
            if all(c in df_filtered.columns for c in ['offer_price', 'stock']):
                df_sample = df_filtered.sample(min(500, len(df_filtered)))
                df_sample['revenue'] = df_sample['offer_price'] * df_sample['stock']

                fig = px.scatter(df_sample, x='offer_price', y='stock',
                               size='revenue', color='revenue',
                               color_continuous_scale='Viridis',
                               labels={'offer_price': 'Price ($)', 'stock': 'Stock', 'revenue': 'Revenue'},
                               hover_data=['title', 'brand'] if 'title' in df_sample.columns else None)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Advanced Price Analysis
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🎯 Average Price by Brand (Top 15)")
            if all(c in df_filtered.columns for c in ['brand', 'offer_price', 'stock']):
                brand_price = df_filtered.groupby('brand').agg({
                    'offer_price': 'mean',
                    'product_id': 'count',
                    'stock': 'sum'
                }).reset_index()
                brand_price.columns = ['brand', 'avg_price', 'nb_products', 'total_stock']
                brand_price['total_value'] = brand_price['avg_price'] * brand_price['total_stock']
                brand_price = brand_price.nlargest(15, 'total_value')

                fig = px.bar(brand_price, y='brand', x='avg_price', orientation='h',
                           color='nb_products', color_continuous_scale='Blues',
                           labels={'avg_price': 'Average Price ($)', 'brand': 'Brand', 'nb_products': 'Nb Products'},
                           hover_data=['total_stock', 'total_value'])
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### 📈 Price-Stock-Revenue Relationship")
            if all(c in df_filtered.columns for c in ['offer_price', 'stock']):
                df_sample = df_filtered.sample(min(500, len(df_filtered)))
                df_sample['revenue'] = df_sample['offer_price'] * df_sample['stock']

                fig = px.scatter(df_sample, x='offer_price', y='stock',
                               size='revenue', color='revenue',
                               color_continuous_scale='Viridis',
                               labels={'offer_price': 'Price ($)', 'stock': 'Stock', 'revenue': 'Revenue'},
                               hover_data=['title', 'brand'] if 'title' in df_sample.columns else None)
                st.plotly_chart(fig, use_container_width=True)

    # ==================== PAGE: RECOMMENDATIONS ====================
    elif page == "🎯 Actionable Recommendations":
        st.title("🎯 Strategic Action Plan")
        st.markdown("### Priority recommendations based on your data")
        st.markdown("---")

        # Calculate impact score for each recommendation
        reco_scores = []

        # Recommendation 1: Restocking
        if 'offer_in_stock' in df_filtered.columns:
            out_count = (df_filtered['offer_in_stock'] == False).sum()
            lost_rev = revenue_metrics.get('lost_revenue', 0)
            if out_count > 0:
                reco_scores.append({
                    'priority': 1,
                    'title': '🔴 URGENT: Restock Out-of-Stock Items',
                    'impact': lost_rev,
                    'effort': 'Medium',
                    'products': out_count,
                    'description': f"Immediately restock {out_count} out-of-stock products",
                    'roi': 'Very High',
                    'timeline': '1-3 days'
                })

        # Recommendation 2: Low stock
        if 'stock' in df_filtered.columns:
            low_stock = df_filtered[(df_filtered['stock'] > 0) & (df_filtered['stock'] < 10)]
            risk_rev = revenue_metrics.get('revenue_at_risk', 0)
            if len(low_stock) > 0:
                reco_scores.append({
                    'priority': 2,
                    'title': '⚠️ IMPORTANT: Boost Low Stock Items',
                    'impact': risk_rev,
                    'effort': 'Medium',
                    'products': len(low_stock),
                    'description': f"Increase stock for {len(low_stock)} products before stockout",
                    'roi': 'High',
                    'timeline': '3-7 days'
                })

        # Recommendation 3: Overstock
        overstock_val = revenue_metrics.get('overstock_value', 0)
        overstock_cnt = revenue_metrics.get('overstock_count', 0)
        if overstock_cnt > 0:
            reco_scores.append({
                'priority': 3,
                'title': '📦 Optimize Overstock',
                'impact': overstock_val * 0.2,
                'effort': 'Low',
                'products': overstock_cnt,
                'description': f"Reduce {overstock_cnt} overstocked items to free up capital",
                'roi': 'Medium',
                'timeline': '2-4 weeks'
            })

        # Recommendation 4: Understocked premium products
        if all(c in df_filtered.columns for c in ['offer_price', 'stock', 'offer_in_stock']):
            premium_low = df_filtered[(df_filtered['offer_price'] > df_filtered['offer_price'].quantile(0.75)) &
                                     (df_filtered['stock'] < 20) &
                                     (df_filtered['offer_in_stock'] == True)]
            if len(premium_low) > 0:
                potential = (premium_low['offer_price'] * (30 - premium_low['stock'])).sum()
                reco_scores.append({
                    'priority': 4,
                    'title': '💎 Boost Premium Products',
                    'impact': potential,
                    'effort': 'High',
                    'products': len(premium_low),
                    'description': f"Increase stock for {len(premium_low)} premium products",
                    'roi': 'Very High',
                    'timeline': '1-2 weeks'
                })

        # Display recommendations
        st.subheader("📋 Priority Action Plan")
        
        for i, reco in enumerate(sorted(reco_scores, key=lambda x: x['impact'], reverse=True), 1):
            with st.expander(f"**#{i} - {reco['title']}** | Impact: ${reco['impact']:,.0f}", expanded=i<=3):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("💰 Revenue Impact", f"${reco['impact']:,.0f}")
                with col2:
                    st.metric("📦 Affected Products", f"{reco['products']}")
                with col3:
                    st.metric("🎯 Expected ROI", reco['roi'])
                with col4:
                    st.metric("⏱️ Timeline", reco['timeline'])

                st.markdown(f"""
                <div class="insight-box">
                <h4>📝 Description</h4>
                <p>{reco['description']}</p>
                <p><strong>Required effort:</strong> {reco['effort']}</p>
                </div>
                """, unsafe_allow_html=True)

                # Specific actions by type
                if 'Out-of-Stock' in reco['title'] or 'Restock' in reco['title']:
                    st.markdown("#### 🎯 Concrete Actions")
                    st.markdown("""
                    1. **Analyze the top 20 products** out of stock by revenue potential
                    2. **Contact suppliers** for express restocking
                    3. **Prioritize** products with the highest sales potential
                    4. **Set up** automatic alerts before stockout
                    """)

                elif 'Low Stock' in reco['title'] or 'Boost Low' in reco['title']:
                    st.markdown("#### 🎯 Concrete Actions")
                    st.markdown("""
                    1. **Identify** products less than 7 days from stockout
                    2. **Increase thresholds** for automatic restocking
                    3. **Negotiate** shorter delivery times
                    4. **Establish** safety stock level by category
                    """)

                elif 'Overstock' in reco['title']:
                    st.markdown("#### 🎯 Concrete Actions")
                    st.markdown("""
                    1. **Analyze** products idle for >90 days
                    2. **Launch** promotions on overstocked items
                    3. **Reduce** future orders for these products
                    4. **Reallocate** capital to high-turnover products
                    """)

                elif 'Premium' in reco['title']:
                    st.markdown("#### 🎯 Concrete Actions")
                    st.markdown("""
                    1. **Increase** stock for high-performing premium products
                    2. **Analyze** historical demand to adjust levels
                    3. **Secure** supply with framework contracts
                    4. **Monitor** these strategic products daily
                    """)
        
        st.markdown("---")

        # Financial Summary
        st.subheader("💰 Optimization Financial Summary")

        total_impact = sum(r['impact'] for r in reco_scores)
        total_products = sum(r['products'] for r in reco_scores)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
            <h3>💰 Total Potential Gain</h3>
            <div class="big-number">${total_impact:,.0f}</div>
            <p>By implementing all recommendations</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
            <h3>📦 Products to Optimize</h3>
            <div class="big-number">{total_products}</div>
            <p>Requiring priority action</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            current_health = inventory_health
            potential_health = min(100, current_health + 25)
            st.markdown(f"""
            <div class="metric-card">
            <h3>📈 Health Improvement</h3>
            <div class="big-number">{current_health:.0f} → {potential_health:.0f}</div>
            <p>Inventory health score</p>
            </div>
            """, unsafe_allow_html=True)

    # ==================== PAGE: ADVANCED ANALYTICS ====================
    elif page == "⚡ Advanced Analytics":
        st.title("⚡ Advanced Analytics & Intelligence")
        st.markdown("### Deep insights and predictive analytics")
        st.markdown("---")

        tab1, tab2, tab3, tab4 = st.tabs(["📊 ABC Analysis", "🎯 Segmentation", "📥 Exports", "🔬 Statistics"])

        with tab1:
            st.subheader("📊 ABC Analysis - Pareto Classification")
            df_abc = analyse_abc(df_filtered)

            if df_abc is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    count_a = (df_abc['ABC_Category'] == 'A').sum()
                    revenue_a = df_abc[df_abc['ABC_Category'] == 'A']['revenue'].sum()
                    pct_a = (revenue_a / df_abc['revenue'].sum() * 100)
                    st.metric("🅰️ Category A", f"{count_a} products ({count_a/len(df_abc)*100:.1f}%)",
                             f"${revenue_a:,.0f} ({pct_a:.1f}%)")
                with col2:
                    count_b = (df_abc['ABC_Category'] == 'B').sum()
                    revenue_b = df_abc[df_abc['ABC_Category'] == 'B']['revenue'].sum()
                    pct_b = (revenue_b / df_abc['revenue'].sum() * 100)
                    st.metric("🅱️ Category B", f"{count_b} products ({count_b/len(df_abc)*100:.1f}%)",
                             f"${revenue_b:,.0f} ({pct_b:.1f}%)")
                with col3:
                    count_c = (df_abc['ABC_Category'] == 'C').sum()
                    revenue_c = df_abc[df_abc['ABC_Category'] == 'C']['revenue'].sum()
                    pct_c = (revenue_c / df_abc['revenue'].sum() * 100)
                    st.metric("©️ Category C", f"{count_c} products ({count_c/len(df_abc)*100:.1f}%)",
                             f"${revenue_c:,.0f} ({pct_c:.1f}%)")

                st.markdown("---")

                col1, col2 = st.columns(2)
                with col1:
                    abc_counts = df_abc['ABC_Category'].value_counts()
                    fig = px.pie(values=abc_counts.values, names=abc_counts.index,
                                title="Product Distribution by Category",
                                color_discrete_map={'A': '#00CC96', 'B': '#FFA15A', 'C': '#EF553B'})
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=list(range(len(df_abc))), y=df_abc['cumulative_pct'],
                                            mode='lines', line=dict(color='#636EFA', width=3),
                                            name='Cumulative Revenue'))
                    fig.add_hline(y=80, line_dash="dash", line_color="red",
                                annotation_text="80% - Category A")
                    fig.add_hline(y=95, line_dash="dash", line_color="orange",
                                annotation_text="95% - Category B")
                    fig.update_layout(title="Pareto Curve",
                                    xaxis_title="Products (sorted by descending revenue)",
                                    yaxis_title="% Cumulative Revenue")
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown(f"""
                <div class="insight-box">
                <strong>💡 ABC Strategy:</strong><br>
                • <strong>Category A ({count_a} products)</strong>: Daily monitoring, never out of stock<br>
                • <strong>Category B ({count_b} products)</strong>: Weekly monitoring, safety stock<br>
                • <strong>Category C ({count_c} products)</strong>: Monthly review, minimal stock
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            st.subheader("🎯 Advanced Product Segmentation")

            if all(c in df_filtered.columns for c in ['offer_price', 'stock']):
                # 4-quadrant segmentation
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

                # Visualization
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
                                   labels={'offer_price': 'Price ($)', 'stock': 'Stock'},
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
                        st.markdown(f"{count} products ({pct:.1f}%)")
                        st.markdown("---")

                # Segment-based recommendations
                st.markdown("### 📋 Actions by Segment")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    <div class="insight-box">
                    <h4>💎 High Value - High Stock</h4>
                    <p><strong>Action:</strong> Excellent! Maintain this level</p>
                    <ul>
                    <li>Monitor daily</li>
                    <li>Ensure turnover</li>
                    <li>Aggressive marketing</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("""
                    <div class="insight-box">
                    <h4>📦 Low Value - High Stock</h4>
                    <p><strong>Action:</strong> Gradually reduce</p>
                    <ul>
                    <li>Promotions to clear</li>
                    <li>Reduce future orders</li>
                    <li>Free up capital</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown("""
                    <div class="alert-critical">
                    <h4>⚠️ High Value - Low Stock</h4>
                    <p><strong>Action:</strong> URGENT - Increase immediately</p>
                    <ul>
                    <li>Restock as priority</li>
                    <li>Increase order thresholds</li>
                    <li>Avoid stockouts at all costs</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("""
                    <div class="insight-box">
                    <h4>🔻 Low Value - Low Stock</h4>
                    <p><strong>Action:</strong> Assess relevance</p>
                    <ul>
                    <li>Analyze profitability</li>
                    <li>Consider discontinuing if unprofitable</li>
                    <li>Or promote if potential</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab3:
            st.subheader("📥 Export Your Reports")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📊 Complete Excel Report")
                st.markdown("Includes: KPIs, filtered data, ABC analysis, recommendations")

                if st.button("📊 Generate Excel Report", type="primary"):
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        # Sheet 1: KPIs
                        kpi_data = pd.DataFrame({
                            'Metric': [
                                'Total Products',
                                'Potential Revenue',
                                'Lost Revenue',
                                '% Lost Revenue',
                                'Revenue at Risk',
                                'Out of Stock Products',
                                'Low Stock Products',
                                'Inventory Health Score',
                                'Total Stock',
                                'Average Price',
                                'Stock Turnover'
                            ],
                            'Value': [
                                int(df_filtered['product_id'].nunique()),
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

                        # Sheet 2: Filtered Data
                        df_filtered.to_excel(writer, sheet_name='Data', index=False)

                        # Sheet 3: ABC Analysis
                        if df_abc is not None:
                            df_abc[['product_id', 'title', 'brand', 'offer_price', 'stock',
                                   'revenue', 'cumulative_pct', 'ABC_Category']].to_excel(
                                writer, sheet_name='ABC Analysis', index=False)

                        # Sheet 4: Stockouts
                        if 'offer_in_stock' in df_filtered.columns:
                            stockouts = df_filtered[df_filtered['offer_in_stock'] == False].copy()
                            stockouts['potential_revenue'] = stockouts['offer_price'] * 10
                            stockouts[['title', 'brand', 'offer_price', 'potential_revenue']].to_excel(
                                writer, sheet_name='Stockouts', index=False)

                    buffer.seek(0)
                    st.download_button(
                        "⬇️ Download Excel Report",
                        buffer,
                        f"complete_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary"
                    )
                    st.success("✅ Report generated successfully!")

            with col2:
                st.markdown("### 📄 CSV Export")
                st.markdown("Filtered data in CSV format for external analysis")

                cols_to_export = st.multiselect(
                    "Columns to export",
                    df_filtered.columns.tolist(),
                    default=[c for c in ['product_id', 'title', 'brand', 'offer_price', 'stock', 'offer_in_stock']
                            if c in df_filtered.columns]
                )

                if cols_to_export and st.button("📄 Generate CSV"):
                    csv = df_filtered[cols_to_export].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Download CSV",
                        csv,
                        f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
                    st.success("✅ CSV generated!")
        
        with tab4:
            st.subheader("🔬 Detailed Statistics")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📊 Price Statistics")
                if 'offer_price' in df_filtered.columns:
                    price_stats = df_filtered['offer_price'].describe()
                    stats_df = pd.DataFrame({
                        'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median (50%)', '75%', 'Max'],
                        'Value': [
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
                st.markdown("#### 📦 Stock Statistics")
                if 'stock' in df_filtered.columns:
                    stock_stats = df_filtered['stock'].describe()
                    stats_df = pd.DataFrame({
                        'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median (50%)', '75%', 'Max'],
                        'Value': [
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

    # ==================== PAGE: BRAND INSIGHTS ====================
    elif page == "🔍 Brand Insights":
        st.title("🔍 Detailed Brand Analysis")
        st.markdown("### Performance and opportunities by brand")
        st.markdown("---")

        if 'brand' in df_filtered.columns:
            # Brand overview
            brand_summary = df_filtered.groupby('brand').agg({
                'product_id': 'count',
                'offer_price': 'mean',
                'stock': ['sum', 'mean']
            }).reset_index()
            brand_summary.columns = ['brand', 'nb_products', 'avg_price', 'total_stock', 'avg_stock']
            brand_summary['total_value'] = brand_summary['avg_price'] * brand_summary['total_stock']
            brand_summary = brand_summary.sort_values('total_value', ascending=False)

            # Top brands
            st.subheader("🏆 Top 15 Brands by Value")
            top_brands = brand_summary.head(15)

            fig = px.bar(top_brands, x='total_value', y='brand', orientation='h',
                        color='nb_products', color_continuous_scale='Viridis',
                        labels={'total_value': 'Total Value ($)', 'brand': 'Brand', 'nb_products': 'Nb Products'},
                        hover_data=['avg_price', 'total_stock'])
            fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Select a brand for detailed analysis
            st.subheader("🔎 Detailed Brand Analysis")
            selected_brand = st.selectbox("Select a brand", sorted(df_filtered['brand'].unique()))

            if selected_brand:
                brand_data = df_filtered[df_filtered['brand'] == selected_brand]

                # Brand KPIs
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("📦 Products", f"{len(brand_data)}")
                with col2:
                    if 'stock' in brand_data.columns:
                        st.metric("📊 Total Stock", f"{brand_data['stock'].sum():,.0f}")
                with col3:
                    if 'offer_price' in brand_data.columns:
                        st.metric("💵 Average Price", f"${brand_data['offer_price'].mean():.2f}")
                with col4:
                    if all(c in brand_data.columns for c in ['offer_price', 'stock']):
                        brand_revenue = (brand_data['offer_price'] * brand_data['stock']).sum()
                        st.metric("💰 Value", f"${brand_revenue:,.0f}")
                with col5:
                    if 'offer_in_stock' in brand_data.columns:
                        availability = ((brand_data['offer_in_stock'] == True).sum() / len(brand_data) * 100)
                        st.metric("✅ Availability", f"{availability:.1f}%")

                st.markdown("---")

                # Brand-specific charts
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 📊 Price Distribution")
                    if 'offer_price' in brand_data.columns:
                        fig = px.histogram(brand_data, x='offer_price', nbins=30,
                                         labels={'offer_price': 'Price ($)', 'count': 'Count'},
                                         color_discrete_sequence=['#636EFA'])
                        st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("#### 📦 Stock Distribution")
                    if 'stock' in brand_data.columns:
                        fig = px.histogram(brand_data, x='stock', nbins=30,
                                         labels={'stock': 'Stock', 'count': 'Count'},
                                         color_discrete_sequence=['#00CC96'])
                        st.plotly_chart(fig, use_container_width=True)
                

    # ==================== PAGE: DATA EXPLORER ====================
    elif page == "📋 Data Explorer":
        st.title("📋 Data Explorer")
        st.markdown("### Detailed and customized view of your data")
        st.markdown("---")

        # Column selection
        cols = df_filtered.columns.tolist()
        defaults = [c for c in ['product_id', 'title', 'brand', 'offer_price', 'stock', 'offer_in_stock'] if c in cols]
        selected_cols = st.multiselect("Select columns to display", cols, default=defaults)

        if selected_cols:
            # Display options
            col1, col2, col3 = st.columns(3)
            with col1:
                sort_col = st.selectbox("Sort by", selected_cols)
            with col2:
                sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)
            with col3:
                rows_to_show = st.number_input("Rows to display", min_value=10, max_value=1000, value=100, step=10)

            # Sort and display
            df_display = df_filtered[selected_cols].copy()
            df_display = df_display.sort_values(sort_col, ascending=(sort_order == "Ascending"))

            st.markdown(f"### Displaying {min(rows_to_show, len(df_display))} rows out of {len(df_display)} total")
            st.dataframe(df_display.head(rows_to_show), use_container_width=True, height=600)

            # Export
            col1, col2 = st.columns(2)
            with col1:
                csv = df_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download as CSV",
                    csv,
                    f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    type="primary"
                )

            with col2:
                st.metric("📊 Total Rows", f"{len(df_display):,}")

else:
    st.warning("📂 Please upload a CSV file using the sidebar to get started.")
    st.info("The CSV file should contain columns like: `title`, `brand`, `offer_price`, `stock`, `offer_in_stock`, `date`.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"📅 Last updated: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
st.sidebar.markdown("🤖 Analytics Dashboard v2.0")
st.sidebar.info("💡 Use filters to refine your analyses")