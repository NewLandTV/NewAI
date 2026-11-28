"""Install as src/commands/newai_ua/__init__.py in Ai-project."""

from newai_ua_bridge import NewAIUABridge
from ua_broadcast import rehearsal


def run():
    try:
        with NewAIUABridge(rehearsal=True) as bridge:
            rehearsal(bridge)
    except (KeyboardInterrupt, EOFError):
        pass
    except (OSError, ValueError, RuntimeError) as error:
        print(f'유아 리허설을 실행하지 못했어요: {error}')
