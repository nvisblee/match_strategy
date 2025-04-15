import streamlit as st
from google.api_core import client_options as client_options_lib
import google.generativeai as genai
import anthropic
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import time
import tempfile

# --- Configuration ---
# NOTE: Use Streamlit secrets for API keys in production!
# Replace with your actual API key or use st.secrets
# GOOGLE_API_KEY = st.secrets["google_api_key"]

GOOGLE_API_KEY = "AIzaSyACUlR4WBNoNtp0zopBQVoUT-GE2H4HCZs"

st.set_page_config(layout="wide", page_title="Tennis Strategy Analyzer")
st.title("🎾 Tennis Point Strategy Analyzer")

# --- API Client Initialization ---
@st.cache_resource
def get_genai_client():
    """Initializes and returns the configured GenAI module."""
    try:
        # Configure the client library with the API key
        genai.configure(
            api_key=GOOGLE_API_KEY,
            transport="rest",
            client_options=client_options_lib.ClientOptions(
                api_endpoint=os.getenv("GOOGLE_API_ENDPOINT"),
            ),
        )
        # Return the configured module itself
        return genai
    except Exception as e:
        st.error(f"Failed to initialize Google GenAI Client: {e}")
        return None

@st.cache_resource
def get_anthropic_client():
    """Initializes and returns the Anthropic client."""
    try:
        # Assumes ANTHROPIC_API_KEY is set as an environment variable ANTHROPIC_API_KEY
        # or replace with your key directly (not recommended).
        # client = anthropic.Anthropic(api_key="YOUR_ANTHROPIC_KEY")
        client = anthropic.Anthropic() # Assumes key is in env var
        return client
    except Exception as e:
        st.error(f"Failed to initialize Anthropic Client: {e}")
        return None

genai_client = get_genai_client()
anthropic_client = get_anthropic_client()

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload a Tennis Point Video (MP4)", type="mp4")

if uploaded_file is not None and genai_client and anthropic_client:
    st.video(uploaded_file)

    # Save the uploaded file temporarily to get a path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_file_path = tmp_file.name

    st.info("Video uploaded. Analyzing... This may take a few moments.")
    analysis_successful = False # Flag to track success

    try:
        # --- 1. Upload file to Google AI ---
        with st.spinner("Uploading file to Google AI..."):
            print("Uploading file...") # Keep console print for reference
            # Use the genai_client module directly for upload_file
            video_file_obj = genai_client.upload_file(path=video_file_path)
            print(f"Completed upload: {video_file_obj.uri}")

            # --- 2. Check file processing status ---
            while video_file_obj.state.name == "PROCESSING":
                print('.', end='') # Keep console print
                st.text(".") # Show progress in streamlit app too
                time.sleep(2) # Slightly longer sleep for web app context
                # Use the genai_client module directly for get_file
                video_file_obj = genai_client.get_file(name=video_file_obj.name)

            if video_file_obj.state.name == "FAILED":
                st.error(f"Video processing failed: {video_file_obj.state.name}")
                st.stop()

            print('Done processing.') # Keep console print
            st.success("File successfully processed by Google AI.")

        # --- 3. Generate Coaching Analysis ---
        with st.spinner("Generating coaching analysis..."):
            coach_prompt = """
You are a professional tennis coach, powered with computer vision, who analyzes strategy for the player the closest to you in the camera angle, allowing image descriptions and other visual tasks. You specializing in analyzing players' match strategy based on the frames that you view. Your expertise includes breaking down a player's gamestyle, shots and shot selection. You provide precise, actionable feedback to help the player improve their gamestyles by analysing specific points. You must pay exceptional attention to detail. Speak directly to the user like a real coach, giving clear, personalized advice.

I will provide a video showcasing a tennis point. You can view and analyze these frames. Your task is to analyze their strategy based on the following aspects:

### Game Style Assessment
- Identify and analyze the player's apparent game style (aggressive baseliner, counterpuncher, all-court player, etc.)
- Look for patterns in their tactical approach
- Evaluate if their chosen tactics aligns with their apparent strengths
- Note if they're effectively executing their game style

### Shot-by-Shot Analysis
Evaluate each shot for:
- Shot characteristics:
  * Spin (topspin, slice, flat)
  * Direction (crosscourt, down the line, inside-out, inside-in)
  * Depth (deep, mid-court, short)
  * Pace and weight of shot
  * try to not encorporate serves in your repsonses as we are only trying to see groundstrokes

- Strategic effectiveness:
  * Was this the right shot selection given the opponent's position and movement?
  * How did it impact the opponent's court coverage and options?
  * Did it create an advantage or surrender one?
  * Specify the opponent's position and movement when analyzing shot choices

### Court Positioning
- Analyze starting position for each shot with exact court references:
  * Distance from baseline
  * Position relative to singles sidelines (centered, specific feet towards deuce/ad side)
  * Distance behind/inside baseline for different shot types
- Evaluate recovery position after shots:
  * Specific recovery location relative to court markers
  * Position relative to opponent's location
- Assess tactical positioning relative to:
  * Opponent's exact court position and movement direction
  * Previous shot selection and court coverage
- If approaching the net, evaluate:
  * Distance from net
  * Position relative to center mark
  * Angle relative to opponent's position

### Critical Moment Analysis
Identify:
- Key tactical decisions that influenced the point, noting exact court positions
- Identify where the shot was hit and where they should have the shot
- Shots that created advantages or surrendered them, with specific positioning details
- Missed opportunities for better tactical choices based on both players' positions
- Crucial positioning decisions with exact court references

### Personalized Feedback Format
1. Game Style Overview
- Identify their game style
- Note how effectively they're implementing it

2. Point Analysis (3-4 key moments)
- Highlight specific shots and decisions with exact court positions
- identify one shot that could have ultimately affected the point, whether that costed them the point or helped them win the point.
Explain the tactical impact of that shot.
- Reference specific court markers and distances

3. Strategic Recommendations (if needed)
- Suggest alternative shots for key moments with specific court positions
- Explain tactical reasoning relative to opponent's location and movement
- Include exact distances and positions for improvements

4. Positive Reinforcement
- Highlight effective tactical decisions with specific examples
- Acknowledge good strategic choices and court positioning

5. Key Takeaway (only if the player needs it, if not, then dont include this section)
- One clear, actionable piece of strategic advice
- Must be specific to their game style and level
- Include specific court positioning references

Remember:
- Be specific with exact court positions and distances
- Always consider and mention opponent's position when analyzing decisions
- Maintain an encouraging but direct coaching voice
- Focus on strategy and decision-making, not technical mechanics
- Personalize advice based on their apparent level and game style
- If they played the point tactically well, acknowledge it and avoid unnecessary suggestions
here are the frames:
"""
            # *** CORRECTED CODE ***
            # Create a model instance using the genai_client module
            model_coach = genai_client.GenerativeModel('models/gemini-2.5-pro-exp-03-25') # Or your preferred model

            # Call generate_content on the model instance
            response_coach = model_coach.generate_content(
                contents=[
                    {"parts": [
                        # Use the processed file object from upload/get_file
                        {"file_data": {"file_uri": video_file_obj.uri, "mime_type": video_file_obj.mime_type}},
                        {"text": coach_prompt}
                    ]}
                ],
                generation_config={"temperature": 0.4} # Added temperature
            )

        st.subheader("Coach's Analysis")
        st.markdown(response_coach.text)

        # --- 4. Identify Key Shot and Positions ---
        with st.spinner("Identifying key shot and positions..."):
            court_prompt = f"""
You are a tennis coach analyzing a video of a point with your student. Your student is the one closest to the camera. Do not tell the timestamp. Please analyze this point and identify:

The first groundstroke (forehand or backhand) the student hit after both players were established in a baseline rally (excluding serves) that either: a) ended the point (winner or error), OR b) seemed to significantly change the rally's dynamic (e.g., created an obvious attacking opportunity for either player, forced a sudden defensive scramble, or resulted in a major shift in court positioning). Identify the shot type (e.g., Student's Forehand Crosscourt creating an attack).
The exact static court positions of both players at the moment the student contacted the ball for this specific identified shot. Be very specific with the position description without using measurements (e.g., "student deep behind the baseline near the center mark," "opponent pulled wide off the ad court").
Based on the positions and the situation during that specific shot identified in #1, suggest a better target or shot direction (e.g., crosscourt deep, down the line, drop shot, lob over opponent at net). Describe where the player hit the shot, from position to where. Now describe where the player should have hit this ball, from postion to where.   
"""
            # *** CORRECTED CODE ***
            # Create a model instance
            model_court = genai_client.GenerativeModel('models/gemini-2.5-pro-exp-03-25')

            # Call generate_content on the model instance
            response_court_text = model_court.generate_content(
                 contents=[
                    {"parts": [
                        {"file_data": {"file_uri": video_file_obj.uri, "mime_type": video_file_obj.mime_type}},
                        {"text": court_prompt}
                    ]}
                ],
                generation_config={"temperature": 0.4} # Added temperature
            )
        st.subheader("Key Moment Identification")
        st.markdown(response_court_text.text)
                # --- 5. Get Visualization Coordinates ---
        with st.spinner("Generating visualization coordinates..."):
            # Define the create_tennis_court function for the prompt context
            # (This is just text for the prompt, not executed here)
            court_template_for_prompt = """
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_tennis_court():
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(5, 10))
    # ... [rest of the template code as provided previously] ...
    ax.set_facecolor('#88b2b9')
    court_width = 36
    court_length = 78
    court = patches.Rectangle((0, 0), court_width, court_length, linewidth=2, edgecolor='white', facecolor='#3a9e5c')
    ax.add_patch(court)
    plt.plot([0, court_width], [0, 0], 'white', linewidth=2)
    plt.plot([0, court_width], [court_length, court_length], 'white', linewidth=2)
    singles_width = 27
    margin = (court_width - singles_width) / 2
    plt.plot([margin, margin], [0, court_length], 'white', linewidth=2)
    plt.plot([court_width - margin, court_width - margin], [0, court_length], 'white', linewidth=2)
    service_line_dist = 21
    plt.plot([margin, court_width - margin], [service_line_dist, service_line_dist], 'white', linewidth=2)
    plt.plot([margin, court_width - margin], [court_length - service_line_dist, court_length - service_line_dist], 'white', linewidth=2)
    plt.plot([court_width / 2, court_width / 2], [service_line_dist, court_length - service_line_dist], 'white', linewidth=2)
    center_mark_width = 0.5
    plt.plot([court_width / 2 - center_mark_width / 2, court_width / 2 + center_mark_width / 2], [0, 0], 'white', linewidth=2)
    plt.plot([court_width / 2 - center_mark_width / 2, court_width / 2 + center_mark_width / 2], [court_length, court_length], 'white', linewidth=2)
    plt.plot([0, court_width], [court_length / 2, court_length / 2], 'white', linewidth=3)
    plt.xlim(-5, court_width + 5)
    plt.ylim(-5, court_length + 5)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('equal')
    return fig, ax
"""
            court_viz_prompt = f"""Based on this analysis of the tennis point:

{response_court_text.text}

Convert the positions and shot direction described above into coordinates that can be used with this tennis court template, but only for the player the closest to you in the camera angle of course:
{court_template_for_prompt}

Provide only:
1. X,Y coordinates for Player 1 position (e.g., [18, 5])
2. X,Y coordinates for Player 2 position (e.g., [18, 73])
3. X,Y coordinates for starting and ending points of the actual shot direction line (e.g., [18, 5] to [10, 70])
4. X,Y coordinates for starting and ending points of the suggested shot direction line (e.g., [18, 5] to [10, 70])

Note:
- Bottom baseline is y=0, top baseline is y=78
- Left singles sideline is x=4.5, right singles sideline is x=31.5
- Net is at y=39
"""
            # *** CORRECTED CODE ***
            # Create a model instance
            model_viz = genai_client.GenerativeModel('models/gemini-2.5-pro-exp-03-25')

            # Call generate_content on the model instance
            response_court_viz_coords = model_viz.generate_content(
                 contents=[
                    # Pass the video file reference again if needed by the model for context,
                    # or just the text prompt if sufficient. Check model requirements.
                    # It's safer to include it if unsure.
                     {"parts": [
                        {"file_data": {"file_uri": video_file_obj.uri, "mime_type": video_file_obj.mime_type}},
                        {"text": court_viz_prompt}
                    ]}
                ],
                generation_config={"temperature": 0.2} # Added temperature
            )
            st.subheader("Visualization Data")
            st.text(response_court_viz_coords.text) # Show the raw coordinates
            
        # --- 6. Generate Visualization Code ---
        with st.spinner("Generating visualization code..."):
            # Define the actual create_tennis_court function for the Anthropic prompt context
            # (Again, this is just text for the prompt)
            create_tennis_court_code_for_prompt = """
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_tennis_court():
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(5, 10))
    ax.set_facecolor('#88b2b9')
    court_width = 36
    court_length = 78
    court = patches.Rectangle((0, 0), court_width, court_length, linewidth=2, edgecolor='white', facecolor='#3a9e5c')
    ax.add_patch(court)
    plt.plot([0, court_width], [0, 0], 'white', linewidth=2)
    plt.plot([0, court_width], [court_length, court_length], 'white', linewidth=2)
    singles_width = 27
    margin = (court_width - singles_width) / 2
    plt.plot([margin, margin], [0, court_length], 'white', linewidth=2)
    plt.plot([court_width - margin, court_width - margin], [0, court_length], 'white', linewidth=2)
    service_line_dist = 21
    plt.plot([margin, court_width - margin], [service_line_dist, service_line_dist], 'white', linewidth=2)
    plt.plot([margin, court_width - margin], [court_length - service_line_dist, court_length - service_line_dist], 'white', linewidth=2)
    plt.plot([court_width / 2, court_width / 2], [service_line_dist, court_length - service_line_dist], 'white', linewidth=2)
    center_mark_width = 0.5
    plt.plot([court_width / 2 - center_mark_width / 2, court_width / 2 + center_mark_width / 2], [0, 0], 'white', linewidth=2)
    plt.plot([court_width / 2 - center_mark_width / 2, court_width / 2 + center_mark_width / 2], [court_length, court_length], 'white', linewidth=2)
    plt.plot([0, court_width], [court_length / 2, court_length / 2], 'white', linewidth=3)
    plt.xlim(-5, court_width + 5)
    plt.ylim(-5, court_length + 5)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('equal')
    return fig, ax
"""
            viz_code_prompt = f"""
generate code to create a pyPlot visualization based on the following coordinate description:
{response_court_viz_coords.text}

Draw Player 1 as a red dot, Player 2 as a blue dot, the actual shot direction as a solid yellow line with an arrowhead at the end, the suggested shot direction as a dashed green line, with an arrowhead at the end.
Use this tennis court template structure:
{create_tennis_court_code_for_prompt}

I need only the Python code block that adds the player dots and the shot line to an existing `ax` object. Assume `fig, ax = create_tennis_court()` has already been called. The code should look like this:

player1_pos = [X, Y] # Extracted from the coordinate description
player2_pos = [X, Y] # Extracted from the coordinate description
shot_start = [X, Y] # Extracted from the coordinate description
shot_end = [X, Y] # Extracted from the coordinate description

# Add Player 1 dot
ax.plot(player1_pos[0], player1_pos[1], 'ro', markersize=8, label='Player 1') # Red dot

# Add Player 2 dot
ax.plot(player2_pos[0], player2_pos[1], 'bo', markersize=8, label='Player 2') # Blue dot

# Add Shot Line
# Use ax.arrow or ax.plot with arrow styling
dx = shot_end[0] - shot_start[0]
dy = shot_end[1] - shot_start[1]
ax.arrow(shot_start[0], shot_start[1], dx, dy, head_width=1, head_length=2, fc='yellow', ec='yellow', linestyle='--', length_includes_head=True, label='Suggested Shot')
# Or: ax.plot([shot_start[0], shot_end[0]], [shot_start[1], shot_end[1]], 'y--', linewidth=2, label='Suggested Shot') # Dashed yellow line

ax.legend() # Add a legend

Provide only the Python code block for plotting these elements, starting from the variable assignments (player1_pos = ...). Do not include the import statements or the function definition.
"""
            # Use the correct model name for Claude 3.5 Sonnet
            message = anthropic_client.messages.create(
                 model="claude-3-5-sonnet-20240620", # Correct model name
                 max_tokens=1024,
                 temperature=0.19,
                 messages=[
                    {
                        "role": "user",
                        "content": viz_code_prompt,
                    }
                ],
            )

            # Extract the code block
            generated_code = ""
            if isinstance(message.content, list) and len(message.content) > 0:
                 code_block = next((block.text for block in message.content if block.type == 'text'), None)
                 if code_block:
                     # Extract code between ```python and ``` or just ``` and ```
                     if "```python" in code_block:
                         generated_code = code_block.split("```python")[1].split("```")[0].strip()
                     elif "```" in code_block:
                          generated_code = code_block.split("```")[1].split("```")[0].strip()
                     else:
                         # Assume the response might just be the code if no backticks
                         # Add basic validation to check if it looks like the expected code start
                         lines = code_block.strip().split('\n')
                         if lines and ('player1_pos =' in lines[0] or 'player_1_pos =' in lines[0]):
                             generated_code = code_block.strip()
                         else:
                             st.warning("Anthropic response format unexpected. Trying to use full response.")
                             generated_code = code_block.strip() # Fallback
                 else:
                    st.warning("Could not find text block in Anthropic response.")
            else:
                st.warning("Anthropic response format unexpected (not a list or empty).")


            

        # --- 7. Execute Visualization Code and Display Plot ---
        if generated_code:
            with st.spinner("Generating visualization..."):
                try:
                    # Define the court creation function locally for execution context
                    def create_tennis_court():
                        # Ensure using plt imported at the top level
                        fig_plot, ax_plot = plt.subplots(figsize=(5, 10))
                        ax_plot.set_facecolor('#88b2b9')
                        court_width = 36
                        court_length = 78
                        # Ensure using patches imported at the top level
                        court = patches.Rectangle((0, 0), court_width, court_length, linewidth=2, edgecolor='white', facecolor='#3a9e5c')
                        ax_plot.add_patch(court)
                        plt.plot([0, court_width], [0, 0], 'white', linewidth=2)
                        plt.plot([0, court_width], [court_length, court_length], 'white', linewidth=2)
                        singles_width = 27
                        margin = (court_width - singles_width) / 2
                        plt.plot([margin, margin], [0, court_length], 'white', linewidth=2)
                        plt.plot([court_width - margin, court_width - margin], [0, court_length], 'white', linewidth=2)
                        service_line_dist = 21
                        plt.plot([margin, court_width - margin], [service_line_dist, service_line_dist], 'white', linewidth=2)
                        plt.plot([margin, court_width - margin], [court_length - service_line_dist, court_length - service_line_dist], 'white', linewidth=2)
                        plt.plot([court_width / 2, court_width / 2], [service_line_dist, court_length - service_line_dist], 'white', linewidth=2)
                        center_mark_width = 0.5
                        plt.plot([court_width / 2 - center_mark_width / 2, court_width / 2 + center_mark_width / 2], [0, 0], 'white', linewidth=2)
                        plt.plot([court_width / 2 - center_mark_width / 2, court_width / 2 + center_mark_width / 2], [court_length, court_length], 'white', linewidth=2)
                        plt.plot([0, court_width], [court_length / 2, court_length / 2], 'white', linewidth=3)
                        plt.xlim(-5, court_width + 5)
                        plt.ylim(-5, court_length + 5)
                        ax_plot.set_xticks([])
                        ax_plot.set_yticks([])
                        plt.axis('equal')
                        return fig_plot, ax_plot

                    # Create the base court
                    fig, ax = create_tennis_court()

                    # Prepare execution environment
                    exec_globals = {"plt": plt, "patches": patches, "fig": fig, "ax": ax}
                    exec_locals = {} # Local scope for execution

                    # Execute the generated code
                    exec(generated_code, exec_globals, exec_locals)

                    # Add title after execution
                    ax.set_title('Key Moment Visualization', color='black', fontsize=16)
                    # Ensure legend is displayed if generated code added labels
                    if ax.get_legend_handles_labels()[0]: # Check if there are any labels
                        ax.legend(loc='best')
                    plt.tight_layout()

                    st.subheader("Shot Visualization")
                    st.pyplot(fig) # Display using Streamlit
                    analysis_successful = True # Mark as successful

                except Exception as e:
                    st.error(f"Error executing visualization code: {e}")
                    st.exception(e) # Show traceback
        else:
            st.warning("No visualization code was generated or extracted.")

    except Exception as e:
        st.error(f"An error occurred during the analysis process: {e}")
        st.exception(e)
    finally:
        # --- Cleanup ---
        # Delete the temporary file
        if 'video_file_path' in locals() and os.path.exists(video_file_path):
            try:
                os.remove(video_file_path)
                print(f"Deleted temporary file: {video_file_path}") # Keep console print
            except Exception as e:
                st.warning(f"Could not delete temporary file {video_file_path}: {e}")

        # Clean up Google AI file if analysis was fully successful (optional)
        # Set delete=True if you want it deleted after successful analysis
        delete_google_file = False
        if analysis_successful and delete_google_file:
            try:
                if 'video_file_obj' in locals() and video_file_obj:
                    print(f"Deleting Google AI file: {video_file_obj.name}")
                    # Use the genai_client module directly for delete_file
                    genai_client.delete_file(name=video_file_obj.name)
                    st.info(f"Cleaned up Google AI file: {video_file_obj.name}")
            except Exception as e:
                st.warning(f"Could not delete Google AI file {video_file_obj.name}: {e}")

elif uploaded_file is None:
    st.info("Please upload a video file to start the analysis.")

else:
    # This part is reached if API client initialization failed
    st.error("API clients could not be initialized. Please check API keys/configurations and restart.")