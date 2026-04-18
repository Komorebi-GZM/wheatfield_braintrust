"""
麦田智囊 CLI工具
命令行接口
"""
import click
import os
from dotenv import load_dotenv

from agents.quick_lesson_prep import QuickLessonPrepAgent
from agents.wisdom_transfer import WisdomTransferAgent
from agents.router import RouterAgent

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
    result = agent.run(subject, grade, topic, context)
    click.echo("\n" + "=" * 50)
    click.echo("生成的教案：")
    click.echo("=" * 50)
    click.echo(result)


@cli.command()
@click.option("--image", "-i", required=True, help="手写教案图片路径")
def wisdom_transfer(image):
    """智慧传承：识别并结构化手写教案"""
    if not os.path.exists(image):
        click.echo(f"错误：文件 {image} 不存在")
        return

    click.echo(f"正在处理 {image} ...")
    agent = WisdomTransferAgent()
    result = agent.run(image)
    click.echo("\n" + "=" * 50)
    click.echo("处理结果：")
    click.echo("=" * 50)
    click.echo(result)


@cli.command()
@click.option("--input", "-i", required=True, help="用户输入")
def route(input):
    """意图路由：识别用户意图"""
    agent = RouterAgent()
    result = agent.run({"user_input": input})
    click.echo(f"\n识别结果：{result.get('intent')}")
    click.echo(f"描述：{result.get('info', {}).get('description', '')}")


@cli.command()
def agents_list():
    """列出所有可用Agent"""
    from agents import __all__ as agent_names
    click.echo("\n可用Agent列表：")
    for name in agent_names:
        click.echo(f"  - {name}")


if __name__ == "__main__":
    cli()
