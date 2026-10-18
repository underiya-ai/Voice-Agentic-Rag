from dotenv import load_dotenv

load_dotenv()  # import se pehle, taaki API keys load ho jayein

from livekit.agents import Agent, AgentServer, AgentSession, TurnHandlingOptions, cli, inference
from livekit.plugins import langchain

from src.livekit_voice_agents.search_agent import Search_agent


class SearchVoiceAgent(Agent):
    """Voice agent: the user asks a question by speaking, the agent searches, and the answer is spoken back."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a voice search assistant. The user asks questions by speaking. "
                "Search when needed, then answer briefly in simple, friendly spoken sentences. "
                "Do not read out links, URLs, or source lists."
            ),
            llm=langchain.LLMAdapter(graph=Search_agent),
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
            "Greet the user briefly and say that you can search and answer their questions "
            "by voice. Ask what they would like to know. Keep it to one or two short sentences."
        )
    )


if __name__ == "__main__":
    cli.run_app(server)