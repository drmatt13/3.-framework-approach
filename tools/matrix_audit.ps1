$ErrorActionPreference = 'Continue'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$outFile = "artifacts\matrix_audit_results.txt"
if (Test-Path $outFile) { Remove-Item $outFile -Force }

$cases = @(
  @{name='matrix2-skl-lr-reg'; lib='scikit-learn'; model='linear_regression'; task='regression'; genArgs=''; runArgs='--save-model=true --verbose=0'},
  @{name='matrix2-skl-log-bin'; lib='scikit-learn'; model='logistic_regression'; task='binary_classification'; genArgs=''; runArgs='--save-model=true --verbose=0 --max-iter=200'},
  @{name='matrix2-skl-log-multi'; lib='scikit-learn'; model='logistic_regression'; task='multiclass_classification'; genArgs=''; runArgs='--save-model=true --verbose=0 --max-iter=200'},
  @{name='matrix2-skl-rf-reg'; lib='scikit-learn'; model='random_forest'; task='regression'; genArgs=''; runArgs='--save-model=true --verbose=0 --n-estimators=50 --cv-folds=2 --cv-n-iter=1'},
  @{name='matrix2-skl-rf-bin'; lib='scikit-learn'; model='random_forest'; task='binary_classification'; genArgs=''; runArgs='--save-model=true --verbose=0 --n-estimators=50 --cv-folds=2 --cv-n-iter=1'},
  @{name='matrix2-skl-rf-multi'; lib='scikit-learn'; model='random_forest'; task='multiclass_classification'; genArgs=''; runArgs='--save-model=true --verbose=0 --n-estimators=50 --cv-folds=2 --cv-n-iter=1'},
  @{name='matrix2-xgb-reg'; lib='xgboost'; model=''; task='regression'; genArgs=''; runArgs='--save-model=true --verbose=0 --n-estimators=50 --cv-folds=2 --cv-n-iter=1'},
  @{name='matrix2-xgb-bin'; lib='xgboost'; model=''; task='binary_classification'; genArgs=''; runArgs='--save-model=true --verbose=0 --n-estimators=50 --cv-folds=2 --cv-n-iter=1'},
  @{name='matrix2-xgb-multi'; lib='xgboost'; model=''; task='multiclass_classification'; genArgs=''; runArgs='--save-model=true --verbose=0 --n-estimators=50 --cv-folds=2 --cv-n-iter=1'},
  @{name='matrix2-tf-reg'; lib='tensorflow'; model='dense_nn'; task='regression'; genArgs='--optimizer=adam --learning_rate=0.001 --epochs=1 --batch_size=32'; runArgs='--save-model=true --verbose=0 --epochs=1 --cv-n-iter=1'},
  @{name='matrix2-tf-bin'; lib='tensorflow'; model='dense_nn'; task='binary_classification'; genArgs='--optimizer=adam --learning_rate=0.001 --epochs=1 --batch_size=32'; runArgs='--save-model=true --verbose=0 --epochs=1 --cv-n-iter=1'},
  @{name='matrix2-tf-multi'; lib='tensorflow'; model='dense_nn'; task='multiclass_classification'; genArgs='--optimizer=adam --learning_rate=0.001 --epochs=1 --batch_size=32'; runArgs='--save-model=true --verbose=0 --epochs=1 --cv-n-iter=1'}
)

"Case|Library|Task|Generate|Run" | Out-File -FilePath $outFile -Encoding utf8

foreach ($c in $cases) {
  $modelPath = Join-Path $repoRoot ("models\{0}.py" -f $c.name)
  $artifactDir = Join-Path $repoRoot ("artifacts\models\{0}" -f $c.name)
  if (Test-Path $modelPath) { Remove-Item $modelPath -Force }
  if (Test-Path $artifactDir) { Remove-Item $artifactDir -Recurse -Force }

  $genCmd = "python tools/generate_model.py --library=$($c.lib) --task=$($c.task) --name=$($c.name)"
  if ($c.model -ne '') { $genCmd += " --model=$($c.model)" }
  if ($c.genArgs -ne '') { $genCmd += " " + $c.genArgs }

  Invoke-Expression $genCmd *> $null
  $genExit = $LASTEXITCODE

  if ($genExit -ne 0) {
    "$($c.name)|$($c.lib)|$($c.task)|FAIL|SKIP" | Out-File -FilePath $outFile -Append -Encoding utf8
    continue
  }

  $runCmd = "python .\models\$($c.name).py $($c.runArgs)"
  Invoke-Expression $runCmd *> $null
  $runExit = $LASTEXITCODE

  $genStatus = if ($genExit -eq 0) { 'PASS' } else { 'FAIL' }
  $runStatus = if ($runExit -eq 0) { 'PASS' } else { 'FAIL' }

  "$($c.name)|$($c.lib)|$($c.task)|$genStatus|$runStatus" | Out-File -FilePath $outFile -Append -Encoding utf8
}

$rows = Get-Content $outFile | Select-Object -Skip 1
$failed = $rows | Where-Object { $_ -match '\|FAIL\|' -or $_ -match '\|SKIP$' }
"FAIL_COUNT=$($failed.Count)" | Out-File -FilePath $outFile -Append -Encoding utf8
"DONE"