"""Exercise release signing validation with real Gradle tasks and temporary keys."""
import os
from pathlib import Path
import secrets
import subprocess
import tempfile

ANDROID = Path(__file__).resolve().parents[1] / 'android'
SIGNING_NAMES = (
    'FEDLEARN_RELEASE_STORE_FILE', 'FEDLEARN_RELEASE_STORE_PASSWORD',
    'FEDLEARN_RELEASE_KEY_ALIAS', 'FEDLEARN_RELEASE_KEY_PASSWORD',
)


def check_case(name, signing, expected_success, expected_message=None):
    env = {k: v for k, v in os.environ.items() if k not in SIGNING_NAMES}
    env.update(signing)
    result = subprocess.run(
        [str(ANDROID / 'gradlew'), '--offline', ':app:preReleaseBuild', '--console=plain'],
        cwd=ANDROID, env=env, capture_output=True, text=True, timeout=180,
    )
    output = result.stdout + result.stderr
    assert (result.returncode == 0) == expected_success, f'{name}: unexpected Gradle exit {result.returncode}'
    if expected_message:
        assert expected_message in output, f'{name}: expected signing refusal was not reported'
    for key in ('FEDLEARN_RELEASE_STORE_PASSWORD', 'FEDLEARN_RELEASE_KEY_PASSWORD'):
        secret = signing.get(key)
        assert not secret or secret not in output, f'{name}: signing password appeared in output'
    print(f'{name}: expected {"success" if expected_success else "refusal"}')


def main():
    check_case('missing credentials', {}, False, 'Release signing requires')
    with tempfile.TemporaryDirectory(prefix='fedlearn-signing-') as directory:
        password = secrets.token_hex(24)
        env = dict(os.environ, FEDLEARN_TEST_PASSWORD=password)
        keytool = Path(os.environ['JAVA_HOME']) / 'bin' / 'keytool'
        paths = {}
        for name, subject in [('release', 'CN=FedLearn Qualification'), ('debug', 'CN=Android Debug,O=Android,C=US')]:
            path = Path(directory) / f'{name}.p12'
            subprocess.run([
                str(keytool), '-genkeypair', '-noprompt', '-alias', 'qualification',
                '-keyalg', 'RSA', '-keysize', '2048', '-validity', '2', '-storetype', 'PKCS12',
                '-keystore', str(path), '-storepass:env', 'FEDLEARN_TEST_PASSWORD',
                '-keypass:env', 'FEDLEARN_TEST_PASSWORD', '-dname', subject,
            ], env=env, check=True, capture_output=True)
            paths[name] = path
        signing = dict(zip(SIGNING_NAMES, [str(paths['release']), password, 'qualification', password]))
        check_case('partial credentials', {SIGNING_NAMES[0]: str(paths['release'])}, False, 'Release signing requires')
        check_case('missing store', {**signing, SIGNING_NAMES[0]: str(Path(directory) / 'absent.p12')}, False, 'Release signing keystore does not exist')
        check_case('wrong store password', {**signing, SIGNING_NAMES[1]: secrets.token_hex(24)}, False, 'Release signing credentials could not open')
        check_case('wrong key alias', {**signing, SIGNING_NAMES[2]: 'absent'}, False, 'Release signing credentials could not open')
        check_case('wrong key password', {**signing, SIGNING_NAMES[3]: secrets.token_hex(24)}, False, 'Release signing credentials could not open')
        check_case('debug certificate', {**signing, SIGNING_NAMES[0]: str(paths['debug'])}, False, 'Android debug certificate cannot sign a release')
        check_case('explicit valid private key', signing, True)
    print('8 real Gradle signing cases passed; temporary qualification keys removed.')


if __name__ == '__main__':
    main()
