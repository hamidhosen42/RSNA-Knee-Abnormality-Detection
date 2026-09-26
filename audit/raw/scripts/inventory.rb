require 'json'; require 'csv'
top = JSON.parse(File.read('top59.json'))
cont = JSON.parse(File.read('containment.json')); names = cont.keys
rows = []
top.each_with_index do |k, i|
  ref = "#{k['author']['userName']}/#{k['currentUrlSlug']}"; base = ref.sub('/', '__')
  meta = JSON.parse(File.read("nb59/#{base}.json"))['metadata'] rescue {}
  vm = JSON.parse(File.read("vm/#{base}.json")) rescue {}
  code = File.read("nb59/#{base}.py")
  src = JSON.parse(File.read("nb59/#{base}.src")) rescue nil
  md = src ? src['cells'].select { |c| c['cell_type'] == 'markdown' }.map { |c| c['source'].is_a?(Array) ? c['source'].join : c['source'] }.join("\n") : ''
  ups = md.scan(%r{kaggle\.com/code/([\w\-]+/[\w\-]+)}).flatten.uniq.reject { |u| u == ref }.first(4)
  ds = k['dataSources'].map { |d| d['dataSourceUrl'] ? d['dataSourceUrl'].sub(%r{^/(datasets|code|models|competitions)/}, '') : "PRIVATE:#{d['reference']['sourceId']}" }
  has = ->(s) { code.include?(s) }
  comps = []
  comps << 'DINOv2S-20' if has['EXPECTED_DINOV2_MEMBERS = 20'] || ds.any? { |d| d.include?('pilkwang/rsna-knee-weights') }
  comps << 'A5/DINOv3S-5fold' if has['m_f0.pt'] || ds.any? { |d| d.include?('knee-mri-fold-weights') || d.include?('dinov3-0917') }
  comps << 'RadImageNet-R50+heads' if has['v52_e11_heads.pt']
  comps << 'Rad-88feat-calibrator' if has['_RAD_CAL_PAYLOAD']
  comps << 'Raptor-CoAtNet' if has['raptor_ft_coatnet_v']
  comps << 'CoAt-resgated' if has['coat_resgated_ep10_top3_manifest']
  comps << 'CoAt-D4depthzone' if has['coatnet_pairfilm_manifest']
  comps << 'CoAt-Global96' if has['global96_top3'] || has['coatnet_global96']
  comps << 'CoAt-RepairV1' if has['repairv1_top3'] || has['coatnet_repairv1']
  comps << 'Raptor-finespacing' if ds.any? { |d| d.include?('finespacing') }
  comps << 'Raptor-widedense' if ds.any? { |d| d.include?('widedense') }
  comps << 'Raptor-arms-x' if ds.any? { |d| d.include?('arms-x') }
  comps << 'band32-leg' if ds.any? { |d| d.include?('band32') }
  comps << 'meniscus-bag' if ds.any? { |d| d.include?('meniscus-bag') }
  comps << 'PRIVATE-leg' if has['OUR LEG'] && ds.any? { |d| d.start_with?('PRIVATE') }
  privn = ds.count { |d| d.start_with?('PRIVATE') }
  fam = if comps.include?('PRIVATE-leg') then 'F0 D4 = F1 + private CNN leg'
        elsif ds.any? { |d| d.include?('dinov3-0917') } then 'F4 Tony-rank bundle (re-hosted public weights)'
        elsif comps.include?('CoAt-Global96') && comps.include?('CoAt-RepairV1') then 'F1a Speedy-Raptors 4-way CoAt (0.943)'
        elsif comps.include?('CoAt-Global96') then 'F1b Speedy-Raptors 3-way CoAt (0.943)'
        elsif comps.include?('CoAt-D4depthzone') then 'F2 0.942 recipe: resgated+D4 CoAt'
        else 'F3 0.941 BTKD/Dinosaurs: resgated-only CoAt' end
  j = names.index(ref)
  nn = names.each_with_index.reject { |b, jj| jj == j }.map { |b, jj| [b, cont[ref][jj]] }.max_by { |x| x[1] }
  g = ->(re) { m = code.match(re); m ? m[1] : '' }
  bestv = vm.dig('bestSubmissionScore', 'kernelVersionNumber'); curv = vm['currentVersionNumber']
  rows << {
    'rank_by_score' => i + 1, 'best_public_score' => k['bestPublicScore'], 'author' => k['author']['userName'], 'slug' => k['currentUrlSlug'],
    'url' => "https://www.kaggle.com/code/#{ref}", 'title' => k['title'],
    'current_version' => curv, 'best_score_version' => bestv, 'best_is_current' => (bestv == curv),
    'total_versions' => vm['totalVersionCount'], 'current_script_version_id' => k['scriptVersionId'],
    'last_run' => k['scriptVersionDateCreated'].to_s[0, 10], 'accelerator' => meta['machineShape'] || (k['isGpuEnabled'] ? 'GPU' : 'CPU'),
    'commit_run_seconds_on_3_visible_studies' => (vm.dig('kernelRun', 'runInfo', 'runTimeSeconds') || '').to_s[0, 7],
    'hidden_test_runtime' => 'NOT EXPOSED BY API', 'license' => 'NOT EXPOSED BY API',
    'votes' => k['totalVotes'], 'forks' => vm.dig('kernel', 'forkCount'), 'medal' => k['medal'],
    'declared_upstreams_in_markdown' => ups.join(';'),
    'nearest_code_neighbor' => nn[0], 'frac_lines_contained_in_neighbor' => nn[1],
    'family' => fam, 'components' => comps.join(';'), 'private_datasources' => privn,
    'rad_exclude' => g[/_RAD_EXCLUDE = (\([^)]*\))/], 'a5_w' => g[/^A5_W = ([0-9.]+)/] .then { |v| v.empty? ? g[/"a5_w": ([0-9.]+)/] : v },
    'rad_alpha' => g[/^_RAD_ALPHA = ([0-9.]+)/].then { |v| v.empty? ? g[/"rad_alpha": ([0-9.]+)/] : v },
    'raptor_view_weights' => code.scan(/"w": (0\.[0-9]+)/).flatten.join('/').then { |v| v.empty? ? g[/"raptor_view_w": (\{[^}]*\})/] : v },
    'coat_private_alpha' => g[/private_alpha = ([0-9.]+)/].then { |v| v.empty? ? g[/"coat_private_alpha": ([0-9.]+)/] : v }[0, 5],
    'ensemble_method' => (code.include?('rank(method') || code.include?('rankpct') ? 'chained percentile-rank blends' : 'see code'),
    'datasets' => ds.reject { |d| d.start_with?('rsna-knee-abnormality-detection') }.join(';'),
    'code_analyzed' => (bestv == curv ? 'current = best-scoring version' : "CURRENT v#{curv}; best-scoring v#{bestv} NOT analyzed")
  }
end
CSV.open('02_public_0941_plus_inventory.csv', 'w') { |csv| csv << rows.first.keys; rows.each { |r| csv << r.values } }
rows.group_by { |r| r['family'] }.each { |f, rs| puts "#{rs.size}\t#{f}\t#{rs.map { |r| r['best_public_score'] }.max}" }
puts rows.count { |r| !r['best_is_current'] }
