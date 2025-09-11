from crewai import Agent, Task, Process, LLM, Crew
import os
from dotenv import load_dotenv

load_dotenv()


# To use Gemini LLM in your Agent
# GEMINI_API_KEY= os.environ.get("GOOGLE_API_KEY")
# model = LLM(
#     model="gemini/gemini-2.0-flash-lite",
#     google_api_key=os.getenv("GOOGLE_API_KEY"),
#     temperature=0.7,
# )

# To use Ollama LLM in your Agent
model = LLM(
    model="ollama/llama3.1:8b",
    base_url="http://localhost:11434"
)



youtube_analyst = Agent(
  role="Youtube Content Analyst",
  goal="Give me a list of 5 most wanted Youtube video ideas about AI , Python and AI agents.",
  backstory="""
  You are an expert at understanding Youtube content trends, audience preferences, and video performance metrics. 
  This is crucial for identifying popular video topics that can attract a wide audience.
  You are good at coming up with creative and engaging video ideas that resonate with viewers.
  """,
  llm=model
)


script_writer = Agent(
  role="Youtube Script Writer",
  goal="Write a detailed Youtube video script about the most wanted Youtube video ideas about AI , Python and AI agents.",
  backstory="""
  You are an expert at writing engaging and informative Youtube video scripts. 
  This is crucial for creating content that captures the audience's attention and keeps them engaged throughout the video.
  You are good at structuring scripts in a way that is easy to follow and understand, while also being entertaining.
  """,
  llm=model
)


task1 = Task(
  description="""Give me a list of 5 most wanted Youtube video ideas about AI , Python and AI agents.
  Write a well detailed report with descriptions of each video idea, target audience, and potential keywords.
  """,
  expected_output="""
  A list of 5 most wanted Youtube video ideas about AI , Python and AI agents.
  A detailed report with descriptions of each video idea, target audience, and potential keywords.
  """,
  agent=youtube_analyst,
)

task2 = Task(
  description="""Write a detailed Youtube video script about the most wanted Youtube video ideas about AI , Python and AI agents.
  The script should be engaging, informative, and easy to follow. It should include an introduction, main content, and conclusion.
  """,
  expected_output="""
  A detailed Youtube video script about the most wanted Youtube video ideas about AI , Python and AI agents.
  The script should be engaging, informative, and easy to follow. It should include an introduction, main content, and conclusion.
  """,
  agent=script_writer,
)

crew = Crew(
  agents=[youtube_analyst, script_writer],
  tasks=[task1, task2],
  process=Process.sequential,
  verbose=True,
)

result = crew.kickoff()

print("Final result:", result)