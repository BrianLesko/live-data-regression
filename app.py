# Brian Lesko 
# 1/21/2024
# Data science studies, real time live data display
# Real time Linear Regression modeling and plotting

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time 
import customize_gui # streamlit GUI modifications
gui = customize_gui.gui()
import plot 
my_plot = plot.plotting()
from sklearn.linear_model import LinearRegression

def main():
    # Set up the app UI
    gui.clean_format(wide=True)
    Sidebar = gui.about(text = "This code implements a live data viewing feed with linear regression predicting the incoming data")
    Title, subTitle, image_spot = st.empty(), st.sidebar.empty(), st.columns([1,5,1])[1].empty()
    with Title:
        st.title("Live Data Feed")

    # Initialize loop variables
    fig, ax = my_plot.get_colored_plt("#F6F6F3",'#335095','#D6D6D6')
    my_plot.set_adaptive_ax(ax)
    # Line 2d objects
    dynamic_plot, = ax.plot([], [], 'o', color='#335095', markersize=5)
    regression_model, = ax.plot([], [], color='red',linewidth=1)
    start_time = time.time()
    # start with some data
    data_log = [(np.array([t/10, np.sin(t/10)]) + np.random.randn(2) * 0.1)[1] for t in range(-10, 1)]
    times = [t/10 for t in range(-10, 1)]
    ax.set_ylim(-1.5, 1.5) # set the Y axis limits to be static

    while True:
        t = time.time() - start_time
        times.append(t)

        # simulate or retrieve data here 
        # noisy sinusoidal data
        data_log.append((np.array([t, np.sin(t)]) + np.random.randn(2) * 0.1)[1])

        # Update the axes limits
        ax.set_xlim(max(times)-10, max(times)+10)
        x_ticks = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 3)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{tick:.2f}' for tick in x_ticks])  # format tick labels

        # linear regresion model
        if len(data_log) > 10:
            times_last_n = np.array(times[-10:]).reshape(-1, 1)
            data_log_last_n = np.array(data_log[-10:]).reshape(-1, 1)
            # Create and fit the model
            model = LinearRegression()
            model.fit(times_last_n, data_log_last_n)

            # Generate x values for the model plot
            x_model = np.linspace(max(times) - .75, max(times) + 1, 2).reshape(-1, 1)
            # Predict y values
            y_model = model.predict(x_model)
            # Plot the model
            regression_model.set_data(x_model, y_model)

        dynamic_plot.set_data(times,data_log)
        with image_spot:
            st.pyplot(fig) 

main()