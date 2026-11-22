"""
麦田智囊 CLI工具
命令行接口
"""

import os

import click
from dotenv import load_dotenv

from maitian_agent.agents.quick_lesson_prep import QuickLessonPrepAgent
from maitian_agent.agents.router import RouterAgent
from maitian_agent.agents.wisdom_transfer import WisdomTransferAgent

load_dotenv()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """麦田智囊 - 乡村教育Agent系统 CLI工具"""
    pass


@cli.command()
@click.option("--subject", "-s", required=True, help="学科")
@click.option("--grade", "-g", required=True, help="年级")
@click.option("--topic", "-t", required=True, help="课题")
@click.option("--context", "-c", default="", help="乡村特色情境")
def lesson_prep(subject, grade, topic, context):
    """极速备课：生成乡土化教案"""
    click.echo(f"正在为 {subject} {grade} 生成教案...")
    agent = QuickLessonPrepAgent()
    try:
        result = agent.run(subject, grade, topic, context)
    except Exception:
        click.echo("服务处理失败，请检查大模型配置后重试", err=True)
        raise click.exceptions.Exit(1) from None
    click.echo("\n" + "=" * 50)
    click.echo("生成的教案：")
    click.echo("=" * 50)
    click.echo(result.data.text)


@cli.command()
@click.option("--image", "-i", required=True, help="手写教案图片路径")
def wisdom_transfer(image):
    """智慧传承：识别并结构化手写教案"""
    if not os.path.exists(image):
        click.echo("错误：指定的图片文件不存在", err=True)
        raise click.exceptions.Exit(1)

    click.echo(f"正在处理 {image} ...")
    agent = WisdomTransferAgent()
    try:
        result = agent.run(image)
    except Exception:
        click.echo("服务处理失败，请检查大模型配置后重试", err=True)
        raise click.exceptions.Exit(1) from None
    click.echo("\n" + "=" * 50)
    click.echo("处理结果：")
    click.echo("=" * 50)
    click.echo(result.data.text)
    if result.status == "needs_review":
        click.echo("待校订：请人工核对识别结果；内容尚未确认或入库。")


@cli.command()
@click.option("--input", "-i", required=True, help="用户输入")
def route(input):
    """意图路由：识别用户意图"""
    agent = RouterAgent()
    try:
        result = agent.run({"user_input": input})
    except Exception:
        click.echo("服务处理失败，请检查大模型配置后重试", err=True)
        raise click.exceptions.Exit(1) from None
    if result.status == "failed":
        click.echo("请求参数无效，请提供非空输入", err=True)
        raise click.exceptions.Exit(1)
    click.echo(f"\n识别结果：{result.data.intent}")
    click.echo(f"描述：{result.data.description}")


@cli.command()
def agents_list():
    """列出所有可用Agent"""
    from maitian_agent.agents import __all__ as agent_names

    click.echo("\n可用Agent列表：")
    for name in agent_names:
        click.echo(f"  - {name}")


if __name__ == "__main__":
    cli()
