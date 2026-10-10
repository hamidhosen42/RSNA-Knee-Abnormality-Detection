# Unmentioned finding = negative, or unknown? (weak labels from report text)

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737650
- **Author:** **Alejandro Zorrilla Bejarano** (@alejandrozobe, Contributor) · _topic author_ · 2026-08-26 15:41 UTC · ▲1
- **Topic votes:** 1 · **Comments/replies:** 1 · **Pinned:** no
- **Last comment:** 2026-08-27 01:58 UTC by Amil Gentili
- **Cloned:** 2026-10-09 09:46 UTC (topic id 737650, raw JSON `_raw/737650.json`)

---

Hi all,

Working on generating weak labels from the report text for the 4,349 unlabelled studies (keyword + negation approach, no LLM). One design question I haven't fully settled:

When a finding (e.g. Fracture) is never mentioned in the report at all, should the weak label be:
(a) 0 (absence) - reasoning: radiologists systematically report what they see, so silence implies absence. Under this closed-world assumption, if a report doesn't mention a finding at all, it's effectively a negative statement by omission - the same way a report doesn't need to say "no headache, shoulder not broken" for every possible unrelated finding; it only mentions what's relevant/present, or
(b) an explicit "unknown/not mentioned" state - reasoning: reports vary a lot in style/completeness across institutions, so silence might just mean the radiologist didn't comment on it.

I saw the report by another user comparing against a published LLM label set (topic 733932/734055) that used an explicit refusal code rather than forcing binary - curious if others have thoughts on which assumption holds better here, especially given how varied report length/style is across the 10+ languages in this dataset.

Any data points (even anecdotal, from manually reading a few reports) would help. Thanks!

---

## Comments (1, threaded, oldest first)

### [1] Amil Gentili

**Amil Gentili** (@agentili, Expert) · 2026-08-27 01:58 UTC · ▲3 · LB rank 488

As a radiologist, if I do not mention a fracture on a knee MRI, it means that I did not see a fracture. I think that most radiologist would do the same for meniscal or fligamentous tear.  For osteoarthtitis it may not be so clear cut, as there are a lot of different way to describe osteoarthritis without mentioning it directly, but by describing it (cartilage loss, chondromalasia, cartilage fissuring, cartilage thinning, osteophytes, etc)
