
# Live Linear Regression
This code implements live data regression for a live data feed written with Streamlit, a low code Web browser UI. This project is written in [Pure Python](https://raw.githubusercontent.com/BrianLesko/live-data-regression/main/app.py) in under 50 lines of code. Created by Brian Lesko for professional engineering purposes.

&nbsp;

<div align="center"><img src="docs/preview.png" width="800"></div>

&nbsp;

## Dependencies

This code uses the following libraries:
- `streamlit`: for building the user interface.
- `numpy`: for creating arrays.
- `matplotlib`: for saving image data


&nbsp;

## Usage

Run the following commands:
```
pip install --upgrade streamlit matplotlib
streamlit run https://raw.githubusercontent.com/BrianLesko/live-data-regression/main/app.py
```

This will start the local Streamlit server, and you can access the chatbot by opening a web browser and navigating to `http://localhost:8501`.

&nbsp;

## Repository Structure
```
repository/
├── app.py # the code and UI integrated together live here
├── customize_gui # class for adding gui elements
├── requirements.txt # the python packages needed to run locally
├── .gitignore # includes the local virtual environment named my_env
├── plot.py # methods used for setting plot axes
├── .streamlit/
│   └── config.toml # theme info for the UI
└── docs/
    └── preview.png # preview photo for Github
```

&nbsp;

## Topics 

This repository was created after my live data feed repository, that plots data in a real time web server window. in that code, I just simulated some noisy sinusoidal data. Challenges there included getting the plot to be stable, as it tended to jiggle when the X axis updated so often. This project came directly after and serves as a review of the basics of machine learning, linear regression. I chose to use scikit learn because this library is most popular for machine learning at the moment. Fully written in python. 

```
scikit-learn | machine learning | linear regression
Python | Low Code | UI | Web application | HTTP | Web server | live data | data science | robotics automation | system management | data collection | monitoring | real time | software dev
Mechanical engineer | Robotics engineer | Engineering
```
&nbsp;

<hr>

&nbsp;

<div align="center">



╭━━╮╭━━━┳━━┳━━━┳━╮╱╭╮        ╭╮╱╱╭━━━┳━━━┳╮╭━┳━━━╮
┃╭╮┃┃╭━╮┣┫┣┫╭━╮┃┃╰╮┃┃        ┃┃╱╱┃╭━━┫╭━╮┃┃┃╭┫╭━╮┃
┃╰╯╰┫╰━╯┃┃┃┃┃╱┃┃╭╮╰╯┃        ┃┃╱╱┃╰━━┫╰━━┫╰╯╯┃┃╱┃┃
┃╭━╮┃╭╮╭╯┃┃┃╰━╯┃┃╰╮┃┃        ┃┃╱╭┫╭━━┻━━╮┃╭╮┃┃┃╱┃┃
┃╰━╯┃┃┃╰┳┫┣┫╭━╮┃┃╱┃┃┃        ┃╰━╯┃╰━━┫╰━╯┃┃┃╰┫╰━╯┃
╰━━━┻╯╰━┻━━┻╯╱╰┻╯╱╰━╯        ╰━━━┻━━━┻━━━┻╯╰━┻━━━╯
  


&nbsp;


<a href="https://twitter.com/BrianJosephLeko"><img src="https://raw.githubusercontent.com/BrianLesko/BrianLesko/f7be693250033b9d28c2224c9c1042bb6859bfe9/.socials/svg-white/x-logo-white.svg" width="30" alt="X Logo"></a> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <a href="https://github.com/BrianLesko"><img src="https://raw.githubusercontent.com/BrianLesko/BrianLesko/f7be693250033b9d28c2224c9c1042bb6859bfe9/.socials/svg-white/github-mark-white.svg" width="30" alt="GitHub"></a> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <a href="https://www.linkedin.com/in/brianlesko/"><img src="https://raw.githubusercontent.com/BrianLesko/BrianLesko/f7be693250033b9d28c2224c9c1042bb6859bfe9/.socials/svg-white/linkedin-icon-white.svg" width="30" alt="LinkedIn"></a>

follow all of these for a cookie :)

</div>


&nbsp;


