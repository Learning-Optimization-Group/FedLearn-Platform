"""Verify actual Android Gradle installation identities without building or installing an APK."""
import json
import os
from pathlib import Path
import subprocess
import tempfile


ANDROID = Path(__file__).resolve().parents[1] / 'android'
INSPECTOR = '''import groovy.json.JsonOutput

gradle.projectsEvaluated {
  def app = gradle.rootProject.findProject(':app')
  if (app != null) app.tasks.register('inspectQualificationPackaging') {
    doLast {
      def android = app.extensions.getByName('android')
      println('FEDLEARN_PACKAGING ' + JsonOutput.toJson([
        applicationId: android.defaultConfig.applicationId,
        versionCode: android.defaultConfig.versionCode,
        versionName: android.defaultConfig.versionName,
        appLabel: android.defaultConfig.manifestPlaceholders['fedlearnAppLabel'],
        releaseDebuggable: android.buildTypes.release.debuggable,
        releaseSigningConfig: android.buildTypes.release.signingConfig?.name
      ]))
    }
  }
}
'''


def main():
    # Configuration inspection needs no signing credentials. Certificate validation has its own gate.
    environment = {key: value for key, value in os.environ.items() if not key.startswith('FEDLEARN_RELEASE_')}
    cases = [
        ('default', [], {'applicationId': 'com.fedlearn.mobile', 'versionCode': 1,
                         'versionName': '2.0.0', 'appLabel': 'FedLearn'}),
        ('explicit production', ['-PfedlearnQualification=false'],
         {'applicationId': 'com.fedlearn.mobile', 'versionCode': 1, 'appLabel': 'FedLearn'}),
        ('qualification update', ['-PfedlearnQualification=true', '-PfedlearnVersionCode=2'],
         {'applicationId': 'com.fedlearn.mobile.qualification', 'versionCode': 2,
          'versionName': '2.0.0-qualification', 'appLabel': 'FedLearn Qualification'}),
        ('maximum version', ['-PfedlearnVersionCode=2100000000'],
         {'applicationId': 'com.fedlearn.mobile', 'versionCode': 2100000000}),
        ('invalid qualification', ['-PfedlearnQualification=garbage'], 'fedlearnQualification must be true or false'),
        ('zero version', ['-PfedlearnVersionCode=0'], 'fedlearnVersionCode must be an integer'),
        ('fractional version', ['-PfedlearnVersionCode=1.5'], 'fedlearnVersionCode must be an integer'),
        ('out-of-range version', ['-PfedlearnVersionCode=2100000001'], 'fedlearnVersionCode must be an integer'),
        ('overflowing version', ['-PfedlearnVersionCode=999999999999999999999'], 'fedlearnVersionCode must be an integer'),
    ]
    with tempfile.TemporaryDirectory(prefix='fedlearn-packaging-') as temporary:
        inspector = Path(temporary) / 'inspect.init.gradle'
        inspector.write_text(INSPECTOR)
        command = [str(ANDROID / 'gradlew'), ':app:inspectQualificationPackaging', '--init-script', str(inspector),
                   '-Dorg.gradle.jvmargs=-Xmx2g -Dfile.encoding=UTF-8', '--no-daemon', '--max-workers=2', '--console=plain']
        for name, properties, expected in cases:
            result = subprocess.run(command + properties, cwd=ANDROID, env=environment,
                                    capture_output=True, text=True, timeout=300)
            lines = [line.removeprefix('FEDLEARN_PACKAGING ') for line in result.stdout.splitlines()
                     if line.startswith('FEDLEARN_PACKAGING ')]
            if isinstance(expected, dict):
                assert result.returncode == 0 and len(lines) == 1, f'{name}: Gradle metadata check failed'
                actual = json.loads(lines[0])
                assert all(actual.get(key) == value for key, value in expected.items()), f'{name}: wrong installation identity'
                assert actual.get('releaseDebuggable') is False, f'{name}: Release is debuggable'
                assert actual.get('releaseSigningConfig') == 'release', f'{name}: release signing gate is missing'
            else:
                assert result.returncode != 0 and not lines, f'{name}: invalid settings were admitted'
                assert expected in result.stdout + result.stderr, f'{name}: expected configuration refusal was not reported'
            print(f'{name}: expected configuration or refusal', flush=True)
    print(f'{len(cases)} actual Gradle packaging checks passed; no APK built or installed.')


if __name__ == '__main__':
    main()
