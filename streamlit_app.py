import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Sales Forecast Dashboard",
    layout="wide"
)

st.title("📈 State Sales Forecast Dashboard")
st.markdown("Forecast next 8 weeks using the best ML model per state")


st.sidebar.header("Controls")

state = st.sidebar.text_input("Enter State Name", "Texas")

forecast_btn = st.sidebar.button("Generate Forecast")



def plot_forecast_chart(df):

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    
    y = df["forecast"] / 1_000_000

    ymin, ymax = y.min(), y.max()
    range_ = ymax - ymin

    
    if range_ < 20:
        step = 2
    elif range_ < 50:
        step = 5
    elif range_ < 100:
        step = 10
    else:
        step = 20

    lower = np.floor(ymin / step) * step
    upper = np.ceil(ymax / step) * step

    ticks = np.arange(lower, upper + step, step)

   
    fig, ax = plt.subplots(figsize=(10,5))

    
    ax.plot(
        df.index,
        y,
        linewidth=2
    )

    
    ax.scatter(
        df.index,
        y,
        color="orange",
        edgecolors="black",
        s=90,
        zorder=3
    )

    ax.set_ylim(lower, upper)
    ax.set_yticks(ticks)

    ax.set_ylabel("Sales (Millions)")
    ax.set_xlabel("Week")

    ax.grid(True)

    st.pyplot(fig)



if forecast_btn:

    with st.spinner("Fetching prediction from backend..."):

        try:
            response = requests.get(f"{API_URL}/forecast/{state}")

            if response.status_code != 200:
                st.error(response.json()["detail"])
            else:

                data = response.json()

                model_used = data["model_used"]
                forecast_df = pd.DataFrame(data["forecast"])

                forecast_df["date"] = pd.to_datetime(forecast_df["date"]).dt.strftime("%d-%m-%Y")

                
                st.success(f"✅ Model Used: {model_used}")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Average (M)",
                    round(forecast_df["forecast"].mean()/1_000_000, 2)
                )

                col2.metric(
                    "Max (M)",
                    round(forecast_df["forecast"].max()/1_000_000, 2)
                )

                col3.metric(
                    "Min (M)",
                    round(forecast_df["forecast"].min()/1_000_000, 2)
                )


                
                st.subheader("📋 Forecast Table")

                st.dataframe(
                    forecast_df,
                    use_container_width=True
                )


                
                csv = forecast_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "⬇ Download CSV",
                    csv,
                    file_name=f"{state}_forecast.csv",
                    mime="text/csv"
                )


              
                st.subheader("📊 Forecast Chart")

                plot_forecast_chart(pd.DataFrame(data["forecast"]))


        except Exception as e:
            st.error(str(e))
