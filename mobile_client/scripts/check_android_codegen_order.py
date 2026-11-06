"""Verify native CMake consumers depend on React Native code-generation producers."""
import json
import os
from pathlib import Path
import subprocess
import tempfile


ANDROID = Path(__file__).resolve().parents[1] / 'android'
INSPECTOR = '''import groovy.json.JsonOutput

gradle.taskGraph.whenReady { graph ->
  if (gradle.rootProject.findProject(':app') == null) return
  def consumers = graph.allTasks.findAll {
    it.project.path == ':app' && it.name.startsWith('configureCMake')
  }
  def producers = graph.allTasks.findAll { it.name == 'generateCodegenArtifactsFromSchema' }
  def missing = []
  consumers.each { consumer ->
    def ancestors = [] as Set
    def visit
    visit = { task ->
      task.taskDependencies.getDependencies(task).each { dependency ->
        if (ancestors.add(dependency.path)) visit(dependency)
      }
    }
    visit(consumer)
    producers.each { producer ->
      if (!ancestors.contains(producer.path)) missing.add([consumer: consumer.path, producer: producer.path])
    }
  }
  println('FEDLEARN_CODEGEN_ORDER ' + JsonOutput.toJson([
    consumers: consumers.collect { it.path },
    producers: producers.collect { it.path },
    missing: missing
  ]))
}
'''


def main():
    environment = {key: value for key, value in os.environ.items() if not key.startswith('FEDLEARN_RELEASE_')}
    with tempfile.TemporaryDirectory(prefix='fedlearn-codegen-order-') as temporary:
        inspector = Path(temporary) / 'inspect.init.gradle'
        inspector.write_text(INSPECTOR)
        command = [str(ANDROID / 'gradlew'), ':app:assembleRelease', '--dry-run', '--init-script', str(inspector),
                   '-Dorg.gradle.jvmargs=-Xmx2g -Dfile.encoding=UTF-8', '--no-daemon', '--max-workers=2', '--console=plain']
        result = subprocess.run(command, cwd=ANDROID, env=environment, capture_output=True, text=True, timeout=300)
    lines = [line.removeprefix('FEDLEARN_CODEGEN_ORDER ') for line in result.stdout.splitlines()
             if line.startswith('FEDLEARN_CODEGEN_ORDER ')]
    diagnostics = '\n'.join((result.stdout + result.stderr).splitlines()[-30:])
    assert result.returncode == 0, f'Gradle task-graph inspection failed with exit {result.returncode}:\n{diagnostics}'
    assert len(lines) == 1, f'Expected one application task-graph record, received {len(lines)}'
    report = json.loads(lines[0])
    assert report['consumers'] and report['producers'], 'Native CMake or code-generation tasks are missing'
    assert not report['missing'], 'CMake is missing code-generation prerequisites: ' + json.dumps(report['missing'])
    print(f"{len(report['consumers'])} CMake configurations depend on all {len(report['producers'])} code-generation tasks.")
    print('Actual dependency edges verified; no APK built or installed.')


if __name__ == '__main__':
    main()
