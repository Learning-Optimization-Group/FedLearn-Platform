"""Check standalone/debug packaging policy against the real configured Gradle project."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

ANDROID = Path(__file__).resolve().parents[1] / 'android'
MARKER = 'FEDLEARN_BUILD_POLICY='
INIT = '''
gradle.projectsEvaluated {
    def app = gradle.rootProject.findProject(':app')
    if (app == null) return
    def android = app.extensions.getByName('android')
    def bundle = app.tasks.findByName('createBundleDebugJsAndAssets')
    def releaseBundle = app.tasks.findByName('createBundleReleaseJsAndAssets')
    def metro = { variant ->
        def value = android.buildTypes.getByName(variant).buildConfigFields['FEDLEARN_METRO_ENABLED']
            ?: android.defaultConfig.buildConfigFields['FEDLEARN_METRO_ENABLED']
        value?.value
    }
    println('FEDLEARN_BUILD_POLICY=' + groovy.json.JsonOutput.toJson([
        debugBundle: bundle != null,
        debugBundleDevelopment: bundle?.devEnabled?.get(),
        debugMetro: metro('debug'),
        releaseBundleDevelopment: releaseBundle?.devEnabled?.get(),
        releaseMetro: metro('release')
    ]))
}
'''


def check(name, properties, expected, init):
    result = subprocess.run(
        [str(ANDROID / 'gradlew'), '--offline', ':app:help', '--console=plain',
         '--max-workers=2', '--init-script', str(init), *properties],
        cwd=ANDROID, env=os.environ.copy(), capture_output=True, text=True, timeout=180,
    )
    output = result.stdout + result.stderr
    if expected is None:
        assert result.returncode != 0 and 'fedlearnStandalone must be true or false' in output, name
    else:
        assert result.returncode == 0, f'{name}: Gradle failed: {output[-3000:]}'
        policies = [line[len(MARKER):] for line in output.splitlines() if line.startswith(MARKER)]
        assert len(policies) == 1, f'{name}: missing configured build policy'
        actual = json.loads(policies[0])
        assert actual == expected, f'{name}: {actual} != {expected}'
    print(f'{name}: passed')


def main():
    with tempfile.TemporaryDirectory(prefix='fedlearn-build-policy-') as directory:
        init = Path(directory) / 'policy.gradle'
        init.write_text(INIT)
        standalone = dict(debugBundle=True, debugBundleDevelopment=True, debugMetro='false',
                          releaseBundleDevelopment=False, releaseMetro='false')
        development = dict(debugBundle=False, debugBundleDevelopment=None, debugMetro='true',
                           releaseBundleDevelopment=False, releaseMetro='false')
        check('standalone embeds Debug policy and disables Metro', ['-PfedlearnStandalone=true'], standalone, init)
        check('ordinary Debug keeps Metro', [], development, init)
        check('explicit false keeps Metro', ['-PfedlearnStandalone=false'], development, init)
        check('invalid mode is refused', ['-PfedlearnStandalone=tru'], None, init)
    print('Four real Gradle packaging-policy cases passed.')


if __name__ == '__main__':
    main()
