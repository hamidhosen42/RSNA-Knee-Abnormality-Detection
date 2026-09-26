require 'json'; require 'set'; require 'digest'
top = JSON.parse(File.read('top59.json'))
refs = top.map { |k| ["#{k['author']['userName']}/#{k['currentUrlSlug']}", k['bestPublicScore']] }
norm = {}
refs.each do |r, s|
  f = "nb59/#{r.sub('/', '__')}.py"
  lines = File.read(f).lines.map { |l| l.sub(/#.*$/, '').strip.gsub(/\s+/, ' ') }
  lines = lines.reject { |l| l.empty? || l.start_with?('#### CELL') || l.length < 8 }
  norm[r] = lines.map { |l| Digest::MD5.hexdigest(l)[0,12] }.to_set
end
# containment: fraction of A's lines present in B
names = refs.map(&:first)
out = []
names.each do |a|
  best = names.reject { |b| b == a }.map { |b| [b, (norm[a] & norm[b]).size.to_f / norm[a].size] }.max_by { |x| x[1] }
  out << [a, norm[a].size, best[0], best[1].round(3)]
end
File.write('containment.json', JSON.dump(names.map { |a| [a, names.map { |b| a==b ? 1.0 : ((norm[a] & norm[b]).size.to_f / [norm[a].size,1].max).round(3) }] }.to_h))
out.each { |o| puts o.join("\t") }
