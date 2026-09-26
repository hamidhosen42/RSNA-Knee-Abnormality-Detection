require 'csv'
T = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
GATE = ['ACL','Medial OA','Lateral OA','PF OA','Effusion',"Baker's",'Contusion']
OUTER = Hash.new(0.60).merge('ACL'=>0.75,'Medial Meniscus'=>0.80,'Lateral Meniscus'=>1.00,'Lateral OA'=>0.75,'Fracture'=>0.75)
def shares(t, rad_exclude, private_w)
  a5w, rad_a, sec, calw = 0.52, 0.55, 0.20, 0.40
  s = Hash.new(0.0)
  base = {'DINOv2S-20'=>1-a5w, 'A5/DINOv3S-5fold'=>a5w}
  cand = if rad_exclude.include?(t) then base.dup
         else base.transform_values { |v| v*(1-rad_a) }.merge('Rad-E10(ref heads)'=>rad_a*0.5, 'Rad-E13'=>rad_a*0.5) end
  branch = cand.transform_values { |v| v*(1-sec) }.merge('Rad-E13-on-E11-layout'=>sec) { |_,a,b| a+b }
  tr = GATE.include?(t) ? branch.transform_values { |v| v*(1-calw) }.merge('Rad-calibrator(88feat,N=4349)'=>calw) : branch
  w = OUTER[t]
  tr.each { |k,v| s[k] += v*(1-w) }
  raptor = {'Raptor maxspan-v5'=>0.60,'Raptor native384dense-v10'=>0.10,'Raptor maxspan-v5-reverse'=>0.10,'Raptor native384-v8'=>0.20}
  raptor.each { |k,v| s[k] += w*0.6*v }
  %w[resgated D4-depthzone Global96 RepairV1].each { |k| s["CoAt #{k}"] += w*0.4*0.25 }
  s = s.transform_values { |v| v*(1-private_w) }
  s['PRIVATE fleet leg'] = private_w if private_w > 0
  s
end
configs = {'Jiwei_v11_0.943'=>[["Baker's",'Fracture'],0.0], 'D4_Lite_0.945'=>[["Baker's"],0.45], 'D4_Blend_0.946'=>[["Baker's"],0.45]}
comps = shares('ACL',[],0.45).keys | shares("Baker's",["Baker's"],0.45).keys | shares('ACL',["Baker's"],0.0).keys
CSV.open('d4_targetwise_analysis.csv','w') do |csv|
  csv << ['config','target','outer_coat_raptor_weight','rad_applied','calibrator_gated'] + comps + ['sum','family_DINO+A5_share','family_RadImageNet_share','family_CoAtNet(Raptor+CoAt)_share','private_share']
  configs.each do |cname,(ex,pw)|
    T.each do |t|
      s = shares(t, ex, pw)
      dino = s.fetch('DINOv2S-20',0)+s.fetch('A5/DINOv3S-5fold',0)
      rad = s.select{|k,_| k.start_with?('Rad')}.values.sum
      coat = s.select{|k,_| k.start_with?('Raptor') || k.start_with?('CoAt')}.values.sum
      csv << [cname,t,OUTER[t],!ex.include?(t),GATE.include?(t)] + comps.map{|c| s.fetch(c, 0.0).round(4)} + [s.values.sum.round(4), dino.round(4), rad.round(4), coat.round(4), s.fetch("PRIVATE fleet leg", 0.0).round(4)]
    end
  end
end
puts "D4 Lite nominal family shares per target:"; T.each { |t| s=shares(t,["Baker's"],0.45); d=s.fetch('DINOv2S-20',0)+s.fetch('A5/DINOv3S-5fold',0); r=s.select{|k,_|k.start_with?('Rad')}.values.sum; c=s.select{|k,_|k.start_with?('Raptor')||k.start_with?('CoAt')}.values.sum; printf("%-17s DINO+A5 %.3f  Rad %.3f  CoAtNet %.3f  private %.3f\n", t, d, r, c, 0.45) }
puts "Fracture: Jiwei vs D4-public-part Rad share:"; p [shares('Fracture',["Baker's",'Fracture'],0.0).select{|k,_|k.start_with?('Rad')}.values.sum.round(4), shares('Fracture',["Baker's"],0.0).select{|k,_|k.start_with?('Rad')}.values.sum.round(4)]
