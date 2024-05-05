from textwrap import dedent
from phi.llm.groq import Groq
from phi.assistant import Assistant

import os
os.environ["GROQ_API_KEY"] = "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"


def get_article_summarizer(
    model: str = "llama3-8b-8192",
    length: int = 500,
    debug_mode: bool = False,
) -> Assistant:
    return Assistant(
        name="Article Summarizer",
        llm=Groq(model=model),
        description="You are a Senior NYT Editor and your task is to summarize a newspaper article.",
        instructions=[
            "You will be provided with the text from a newspaper article.",
            "Carefully read the article a prepare a thorough report of key facts and details.",
            f"Your report should be less than {length} words.",
            "Provide as many details and facts as possible in the summary.",
            "Your report will be used to generate a final New York Times worthy report.",
            "REMEMBER: you are writing for the New York Times, so the quality of the report is important.",
            "Make sure your report is properly formatted and follows the <report_format> provided below.",
        ],
        add_to_system_prompt=dedent("""
        <report_format>
        **Overview:**\n
        {overview of the article}

        **Details:**\n
        {details/facts/main points from the article}

        **Key Takeaways:**\n
        {provide key takeaways from the article}
        </report_format>
        """),
        # This setting tells the LLM to format messages in markdown
        markdown=True,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,

    )


def get_thread_writer(
    model: str = "llama3-70b-8192",
    debug_mode: bool = False,
) -> Assistant:
    return Assistant(
        name="Twitter thread writer",
        llm=Groq(model=model),
        description="You are a Senior Twitter account manager and your task "
                    "is to write a award winning Twitter worthy thread due tomorrow.",
        instructions=[
            "You will be provided with a topic and pre-processed summaries from junior researchers.",
            "Carefully read the provided information and think about the contents",
            "Then generate a award winning Twitter worthy thread in the <article_format> provided below.",
            "Make your twitter thread engaging, informative, and well-structured.",
            "Break the thread into sections and provide hashtags only in the first tweet.",
            "Make sure the intro is catchy and engaging.",
            "REMEMBER: you are writing for an award winning IT company, so the quality of the thread is important.",
        ],
        add_to_system_prompt=dedent("""
        <thread_format>

        <tweet_1>
        {intro tweet under 240 characters}
        </tweet_1>

        <tweet_2>
        {engaging followup tweet under 240 characters}
        </tweet_2>

        </thread_format>
        """),

        # This setting tells the LLM to format messages in markdown
        markdown=True,
        add_datetime_to_instructions=False,
    )