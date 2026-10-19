from dotenv import load_dotenv

load_dotenv()  # import se pehle, taaki API keys load ho jayein
from src.livekit_voice_agents.Rag.config import settings
from livekit.agents import Agent, AgentServer, AgentSession, TurnHandlingOptions, cli, inference
from livekit.plugins import langchain

from src.livekit_voice_agents.search_agent import Policy_agent


class SearchVoiceAgent(Agent):
    """Voice agent: the user asks a question by speaking, the agent searches, and the answer is spoken back."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a voice assistant for employees that answers questions about company policy. "
                "The employee asks questions by speaking. "
                "For company rules, leave, refunds, procedures, or HR policies, search the company policy first. "
               "If the policy does not have the answer, or the question is about external facts or real-time information, search the web. "
               "Answer briefly in simple, friendly spoken sentences, and mention the page number when the answer comes from the policy. "
               "If you cannot find the answer, say so honestly and suggest contacting HR or the manager. "
               "Do not read out links, URLs, or source lists."
            ),
            llm=langchain.LLMAdapter(graph=Policy_agent),
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx):
    session = AgentSession(
        stt=inference.STT(language="multi"),
        tts=inference.TTS(model="inworld/inworld-tts-2", voice="Ashley"),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            endpointing={"min_delay": 0.5, "max_delay": 15.0},
        ),
    )

    await session.start(agent=SearchVoiceAgent(), room=ctx.room)

    await session.generate_reply(
        instructions=(
            "Greet the employee in one short sentence. Say you can answer company policy and general questions. Ask what they need."
        )
    )


if __name__ == "__main__":
    cli.run_app(server)