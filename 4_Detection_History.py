import streamlit as st
import json
import pandas as pd

st.title("📜 Detection History")

try:
    with open("detection_history.json", "r") as f:
        history = json.load(f)

    if not history:
        st.warning("Detection history is empty.")
    else:
        # Print sample keys to debug
        st.subheader("Sample Entry Keys")
        st.json(history[0])  # show keys of first detection

        # Convert to DataFrame
        df = pd.DataFrame(history)

        # Show all available columns
        st.subheader("Detection History Table")
        st.dataframe(df, use_container_width=True)

        # OPTIONAL: Try parsing timestamp if it exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values("timestamp", ascending=False)
except FileNotFoundError:
    st.error("No detection history file found.")
except Exception as e:
    st.error(f"Error: {str(e)}")
