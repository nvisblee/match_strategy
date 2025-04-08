from IPython.display import display, Image, Audio
import cv2
import base64
import anthropic
from IPython.display import Markdown
from google import genai
from openai import OpenAI
import os
import time





import os


client = genai.Client(api_key="AIzaSyACUlR4WBNoNtp0zopBQVoUT-GE2H4HCZs")



print("Uploading file...")
video_file = client.files.upload(file="tennisSix.mp4")
print(f"Completed upload: {video_file.uri}")



# Check whether the file is ready to be used.
while video_file.state.name == "PROCESSING":
    print('.', end='')
    time.sleep(1)
    video_file = client.files.get(name=video_file.name)

if video_file.state.name == "FAILED":
  raise ValueError(video_file.state.name)

print('Done')

# Pass the video file reference like any other media part.
response = client.models.generate_content(
    model="models/gemini-2.0-flash",
    contents=[
        {"parts": [
            {"file_data": {"file_uri": video_file.uri}},
            {"text": """           
You are a professional tennis coach, powered with computer vision, allowing image descriptions and other visual tasks. You specializing in analyzing players' match strategy based on the frames that you view. Your expertise includes breaking down a player's gamestyle, shots and shot selection. You provide precise, actionable feedback to help the player improve their gamestyles by analysing specific points. You must pay exceptional attention to detail. Speak directly to the user like a real coach, giving clear, personalized advice.

I will provide a video ( showcasing a tennis point. You can view and analyze these frames. Your task is to analyze their strategy based on the following aspects:

### Game Style Assessment
- Identify and analyze the player's apparent game style (aggressive baseliner, counterpuncher, all-court player, etc.)
- Look for patterns in their tactical approach
- Evaluate if their chosen tactics align with their apparent strengths
- Note if they're effectively executing their game style

### Shot-by-Shot Analysis
Evaluate each shot for:
- Shot characteristics:
  * Spin (topspin, slice, flat)
  * Direction (crosscourt, down the line, inside-out, inside-in)
  * Depth (deep, mid-court, short)
  * Pace and weight of shot

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
"""} 
        ]}
    ])

# Print the response, rendering any Markdown
print(response.text)


#This call can definitely be split into two calls''
court = client.models.generate_content(
    model="models/gemini-2.0-flash",
    contents=[
        {"parts": [
            {"file_data": {"file_uri": video_file.uri}},
            {"text": f"""You are powered with computer vision, allowing image descriptions and other visual tasks. You can directly view and analyze frames of a tennis point. Please analyze the frames of the point and:

Please identify:
1. identify the groundstroke (no serves) that influenced the point the most. 
2. The exact static court positions of both players during this shot. The players must be on the opposite sides of the court, one player on one side and the other player on the other side. Be very specific with the position description without using measurements (like right behind baseline for example).
3. The direction of where the player should have hit the shot. Be descriptive (cross court, down the line, down the middle etc).
"""
            }
        ]}
    ]
)
print(court.text)

# Get the tennis court visualization information
court_viz = client.models.generate_content(
    model="models/gemini-2.0-flash",
    contents=[
        {"parts": [
            {"file_data": {"file_uri": video_file.uri}},
            {"text": f"""Based on this analysis of the tennis point:

{court.text}

Convert the positions and shot direction described above into coordinates that can be used with this tennis court template:

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_tennis_court():
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(5, 10))
   
    # Set background color
    ax.set_facecolor('#88b2b9')  # Light blue-green for background
   
    # Tennis court dimensions (in feet)
    court_width = 36  # Width of the court (including doubles alley)
    court_length = 78  # Length of the court
   
    # Draw the court outline - for vertical orientation
    court = patches.Rectangle((0, 0), court_width, court_length,
                            linewidth=2, edgecolor='white', facecolor='#3a9e5c')
    ax.add_patch(court)
   
    # Draw the baselines (top and bottom)
    plt.plot([0, court_width], [0, 0], 'white', linewidth=2)  # Bottom baseline
    plt.plot([0, court_width], [court_length, court_length], 'white', linewidth=2)  # Top baseline
   
    # Draw the singles sidelines (left and right)
    singles_width = 27  # Singles court width
    margin = (court_width - singles_width) / 2
    plt.plot([margin, margin], [0, court_length], 'white', linewidth=2)  # Left singles sideline
    plt.plot([court_width - margin, court_width - margin], [0, court_length], 'white', linewidth=2)  # Right singles sideline
   
    # Draw the service lines
    service_line_dist = 21  # Distance from baseline to service line
    plt.plot([margin, court_width - margin], [service_line_dist, service_line_dist], 'white', linewidth=2)  # Bottom service line
    plt.plot([margin, court_width - margin], [court_length - service_line_dist, court_length - service_line_dist], 'white', linewidth=2)  # Top service line
   
    # Draw the center service line
    plt.plot([court_width / 2, court_width / 2], [service_line_dist, court_length - service_line_dist], 'white', linewidth=2)  # Center service line
   
    # Draw the center mark on the baselines
    center_mark_width = 0.5
    plt.plot([court_width / 2 - center_mark_width / 2, court_width / 2 + center_mark_width / 2], [0, 0], 'white', linewidth=2)  # Bottom center mark
    plt.plot([court_width / 2 - center_mark_width / 2, court_width / 2 + center_mark_width / 2], [court_length, court_length], 'white', linewidth=2)  # Top center mark
   
    # Draw the net
    plt.plot([0, court_width], [court_length / 2, court_length / 2], 'white', linewidth=3)  # Net
   
    # Set axis limits
    plt.xlim(-5, court_width + 5)
    plt.ylim(-5, court_length + 5)
   
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
   
    # Add court labels
    plt.text(court_width / 2, -3, 'Baseline', color='white', ha='center')
    plt.text(court_width / 2, court_length + 3, 'Baseline', color='white', ha='center')
    plt.text(-3, court_length / 2, 'Net', color='white', va='center', rotation=90)
    plt.text(court_width + 3, service_line_dist / 2, 'Service Box', color='white', va='center', rotation=90)
    plt.text(court_width + 3, court_length - service_line_dist / 2, 'Service Box', color='white', va='center', rotation=90)
   
    # Set title
    plt.title('Tennis Court Diagram', color='black', fontsize=16)
   
    # Set equal aspect ratio
    plt.axis('equal')
   
    # Show the plot
    plt.tight_layout()
    return fig, ax

Provide only:
1. X,Y coordinates for Player 1 position (e.g., [18, 5])
2. X,Y coordinates for Player 2 position (e.g., [18, 73])
3. X,Y coordinates for starting and ending points of the shot direction line (e.g., [18, 5] to [10, 70])

Note:
- Bottom baseline is y=0, top baseline is y=78
- Left sideline is x=4.5, right sideline is x=31.5
- Net is at y=39
"""
            }
        ]}
    ]
)
print(court_viz.text)



client_claude = anthropic.Anthropic()
message = client_claude.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=2048,
    temperature=0.2,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""           
generate code to create a pyPlot visualization based on the following description: 
{court_viz.text} 
Draw a line visualizing where the shot should have been hit. Use the tennis court template:
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_tennis_court():
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(5, 10))
   
    # Set background color
    ax.set_facecolor('#88b2b9')  # Light blue-green for background
   
    # Tennis court dimensions (in feet)
    court_width = 36  # Width of the court (including doubles alley)
    court_length = 78  # Length of the court
   
    # Draw the court outline - for vertical orientation
    court = patches.Rectangle((0, 0), court_width, court_length,
                            linewidth=2, edgecolor='white', facecolor='#3a9e5c')
    ax.add_patch(court)
   
    # Draw the baselines (top and bottom)
    plt.plot([0, court_width], [0, 0], 'white', linewidth=2)  # Bottom baseline
    plt.plot([0, court_width], [court_length, court_length], 'white', linewidth=2)  # Top baseline
   
    # Draw the singles sidelines (left and right)
    singles_width = 27  # Singles court width
    margin = (court_width - singles_width) / 2
    plt.plot([margin, margin], [0, court_length], 'white', linewidth=2)  # Left singles sideline
    plt.plot([court_width - margin, court_width - margin], [0, court_length], 'white', linewidth=2)  # Right singles sideline
   
    # Draw the service lines
    service_line_dist = 21  # Distance from baseline to service line
    plt.plot([margin, court_width - margin], [service_line_dist, service_line_dist], 'white', linewidth=2)  # Bottom service line
    plt.plot([margin, court_width - margin], [court_length - service_line_dist, court_length - service_line_dist], 'white', linewidth=2)  # Top service line
   
    # Draw the center service line
    plt.plot([court_width / 2, court_width / 2], [service_line_dist, court_length - service_line_dist], 'white', linewidth=2)  # Center service line
   
    # Draw the center mark on the baselines
    center_mark_width = 0.5
    plt.plot([court_width / 2 - center_mark_width / 2, court_width / 2 + center_mark_width / 2], [0, 0], 'white', linewidth=2)  # Bottom center mark
    plt.plot([court_width / 2 - center_mark_width / 2, court_width / 2 + center_mark_width / 2], [court_length, court_length], 'white', linewidth=2)  # Top center mark
   
    # Draw the net
    plt.plot([0, court_width], [court_length / 2, court_length / 2], 'white', linewidth=3)  # Net
   
    # Set axis limits
    plt.xlim(-5, court_width + 5)
    plt.ylim(-5, court_length + 5)
   
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
   
    # Add court labels
    plt.text(court_width / 2, -3, 'Baseline', color='white', ha='center')
    plt.text(court_width / 2, court_length + 3, 'Baseline', color='white', ha='center')
    plt.text(-3, court_length / 2, 'Net', color='white', va='center', rotation=90)
    plt.text(court_width + 3, service_line_dist / 2, 'Service Box', color='white', va='center', rotation=90)
    plt.text(court_width + 3, court_length - service_line_dist / 2, 'Service Box', color='white', va='center', rotation=90)
   
    # Set title
    plt.title('Tennis Court Diagram', color='black', fontsize=16)
   
    # Set equal aspect ratio
    plt.axis('equal')
   
    # Show the plot
    plt.tight_layout()
    return fig, ax

# Create and show the tennis court
fig, ax = create_tennis_court()
plt.show()


I need only the code, nothing else. 
""",
                }
            ],
        }
    ],
)
print(message.content)