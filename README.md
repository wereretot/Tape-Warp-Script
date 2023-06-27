# Music Warper

Music Warper is a Python script that simulates the effect of a worn-out cassette tape on music files. It randomly changes the speed and pitch of the input music file, creating a nostalgic and warped audio experience.

## Features

- Randomly alters the speed and pitch of the music file within a specified range.
- Mimics the characteristics of a worn-out cassette tape, including speed instability and pitch warping.
- Configurable settings to adjust the speed range, ambient temperature, and wobble factor for customization.

## Prerequisites

- Python 3.x
- PyDub library: `pip install pydub`
- tqdm library: `pip install tqdm`
- Tkinter library (usually included in Python installations)

## Usage

1. Clone or download the Music Warper repository.

2. Install the required libraries mentioned in the Prerequisites section, if not already installed.

3. Configure the settings in the `config.ini` file:

   - `speed_min`: The minimum speed factor for warping (default: 0.98)
   - `speed_max`: The maximum speed factor for warping (default: 1.02)
   - `ambient_temperature`: The ambient temperature at the time of playback in degrees Celsius (default: 25)
   - `wobble_factor`: The probability of introducing wobble effect (default: 0.3)
    
4. Run the script with 'python3 speed.py'.

5. Navigate to the music file you want to warp. Don't worry if it's not in mpeg3 format, the script will automatically convert it to it and start the conversion automatically.

Written by ChatGPT. Checked and verified by @wereretot
