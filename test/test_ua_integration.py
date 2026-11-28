import importlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, call, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'integrations/ai-project'))

from install import install
from newai_ua_bridge import NewAIUABridge
from ua_broadcast import follow, present, rehearsal, viewer_question


class BroadcastTests(unittest.TestCase):
    def setUp(self):
        self.bridge = Mock()
        self.bridge.get_response.return_value = ['유아의', ' 답변\x1b\u202e']
        self.obs = Mock()

    def test_only_opt_in_chat_is_considered(self):
        for text in ('안녕하세요', '!ua', '/run newai_ua', '!ua /run games.worch', '!ua !help', None):
            self.assertEqual(viewer_question(text), '')
        self.assertEqual(viewer_question('!ua   방송 인사해 줘!'), '방송 인사해 줘!')
        self.assertEqual(len(viewer_question('!ua ' + '가' * 500)), 300)

    def test_approval_precedes_subtitles_and_voice_and_filters_controls(self):
        approve = Mock(return_value=False)
        self.assertFalse(present(self.bridge, self.obs, '인사', approve, voice=True))
        approve.assert_called_once_with('유아의 답변')
        self.obs.update_obs_text.assert_not_called()
        self.bridge.read_aloud.assert_not_called()

        outputs = Mock()
        outputs.attach_mock(self.obs.update_obs_text, 'subtitle')
        outputs.attach_mock(self.bridge.read_aloud, 'speak')
        outputs.approve.return_value = True
        self.assertTrue(present(self.bridge, self.obs, '인사', outputs.approve, voice=True))
        self.assertEqual(outputs.mock_calls, [call.approve('유아의 답변'), call.subtitle('Answer Text', '유아의 답변'),
                                              call.speak('유아의 답변')])

    def test_empty_reply_does_not_request_approval(self):
        self.bridge.get_response.return_value = ['\n\u202e']
        approve = Mock()
        self.assertFalse(present(self.bridge, self.obs, '인사', approve, voice=True))
        approve.assert_not_called()
        self.obs.update_obs_text.assert_not_called()

    def test_voice_only_rehearsal_still_requires_approval(self):
        with patch('builtins.input', side_effect=['인사', '', '/reset', '/exit']), patch('builtins.print'):
            rehearsal(self.bridge, voice=True)
        self.bridge.read_aloud.assert_not_called()
        self.bridge.reset.assert_called_once()

    def test_rehearsal_keeps_running_after_a_failed_generation(self):
        self.bridge.get_response.side_effect = [RuntimeError('generation failed'), ['다시 반가워요!']]
        with patch('builtins.input', side_effect=['첫 질문', '두 번째 질문', '공개', '/exit']), patch('builtins.print'):
            rehearsal(self.bridge, self.obs)
        self.obs.update_obs_text.assert_called_once_with('Answer Text', '다시 반가워요!')

    def test_live_relay_uses_latest_eligible_message_and_approval(self):
        chat = Mock()
        chat.is_alive.side_effect = [True, False]
        chat.get.return_value.sync_items.return_value = [Mock(message='!ua 이전 질문'),
                                                         Mock(message='!ua 새 질문'), Mock(message='/run math.add')]
        with patch('ua_broadcast.sleep'), patch('builtins.input', return_value='공개'), patch('builtins.print'):
            follow(chat, self.bridge, self.obs, voice=True)
        self.bridge.get_response.assert_called_once_with('새 질문')
        self.obs.update_obs_text.assert_called_once_with('Answer Text', '유아의 답변')


class InstallationTests(unittest.TestCase):
    def fixture(self, folder):
        root = Path(folder)
        (root / 'src/commands').mkdir(parents=True)
        (root / 'src/commands/__init__.py').write_text('# original command loader\n')
        (root / 'src/chatbot.py').write_text('# original chatbot\n')
        return root

    def test_additive_install_and_command_import(self):
        with TemporaryDirectory() as folder:
            root = self.fixture(folder)
            install(root)
            install(root)
            self.assertEqual((root / 'src/chatbot.py').read_text(), '# original chatbot\n')
            sys.path.insert(0, str(root / 'src'))
            try:
                command = importlib.import_module('commands.newai_ua')
                with patch.object(command, 'NewAIUABridge'), patch.object(command, 'rehearsal') as run:
                    command.run()
                    run.assert_called_once()
            finally:
                sys.path.pop(0)
                sys.modules.pop('commands.newai_ua', None)
                sys.modules.pop('commands', None)

    def test_conflict_leaves_existing_files_unchanged(self):
        with TemporaryDirectory() as folder:
            root = self.fixture(folder)
            relay = root / 'src/ua_broadcast.py'
            relay.write_text('# existing relay\n')
            with self.assertRaises(FileExistsError):
                install(root)
            self.assertEqual(relay.read_text(), '# existing relay\n')
            self.assertFalse((root / 'src/newai_ua_bridge.py').exists())


class BridgeTests(unittest.TestCase):
    def test_external_worker_protocol_and_environment(self):
        script = '''
import json, os, sys
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
assert 'DISCORD_BOT_TOKEN' not in os.environ
assert 'obs_password' not in os.environ
print('{"type":"ready"}', flush=True)
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({'type': 'answer', 'text': request['prompt'] + ' · 유아'}, ensure_ascii=False), flush=True)
'''
        with patch.dict('os.environ', {'NEWAI_ROOT': str(ROOT), 'DISCORD_BOT_TOKEN': 'fake', 'obs_password': 'fake'}):
            with NewAIUABridge(worker_command=[sys.executable, '-u', '-c', script]) as bridge:
                self.assertEqual(list(bridge.get_response('반가워요')), ['반가워요 · 유아'])
                proc = bridge.worker.proc
            self.assertIsNotNone(proc.poll())

    def test_private_history_is_bounded_and_live_has_no_history(self):
        for rehearsal_mode in (True, False):
            with self.subTest(rehearsal=rehearsal_mode), patch.dict('os.environ', {'NEWAI_ROOT': str(ROOT)}):
                with NewAIUABridge(rehearsal=rehearsal_mode) as bridge:
                    bridge.worker = Mock()
                    bridge.worker.exchange.return_value = '유아의 답변이에요.'
                    for i in range(5):
                        self.assertEqual(list(bridge.get_response(f'인사 {i}')), ['유아의 답변이에요.'])
                    request = bridge.worker.exchange.call_args.args[0]
                    self.assertEqual([pair['user'] for pair in request['history']],
                                     ['인사 1', '인사 2', '인사 3'] if rehearsal_mode else [])
                    bridge.last_turn -= 901
                    list(bridge.get_response('오랜만이에요'))
                    self.assertEqual(bridge.worker.exchange.call_args.args[0]['history'], [])
                    bridge.reset()
                    self.assertEqual(list(bridge.history), [])


if __name__ == '__main__':
    unittest.main()
