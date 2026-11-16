# TuneJudge knowledge base

Short notes the RAG retriever searches. Each `## kind: name` section is one
document. Energy and acoustic numbers are averages from data/songs.csv
(0 = none, 1 = maximum). Keep each note focused on words a listener might
type, so keyword retrieval can find it.

## genre: pop
Catchy, polished songs with big hooks and choruses, made for radio and singing along. Average energy 0.59, acoustic 0.30. Good for parties, driving, getting ready, upbeat everyday listening. Related genre: indie pop.

## genre: indie pop
Pop with a homemade, dreamy or quirky feel: jangly guitars, soft synths, bedroom production. Average energy 0.57, acoustic 0.28. Good for road trips, sunny afternoons, coffee shops, nostalgic moods. Related genre: pop.

## genre: lofi
Lo-fi hip hop beats: dusty drums, warm piano, vinyl crackle, mostly instrumental. Average energy 0.37, acoustic 0.58. The classic choice for studying, homework, reading, focus, coding, rainy days and late nights. Related genre: ambient.

## genre: ambient
Slow, spacious, atmospheric sound with soft pads and almost no beat. Average energy 0.17, acoustic 0.85. Good for sleep, meditation, relaxing, background focus, calm and peaceful moments. Related genre: lofi.

## genre: rock
Guitar-driven bands with drums and strong vocals, from classic rock to alternative and grunge. Average energy 0.68, acoustic 0.17. Good for driving, workouts, cleaning, feeling rebellious or pumped up.

## genre: metal
Heavy distorted guitars, fast drums and powerful or screamed vocals. Average energy 0.82, acoustic 0.05, the loudest genre in the catalog. Good for intense workouts, lifting weights, venting anger, aggressive moods.

## genre: jazz
Swing, bebop, cool jazz and vocal standards: saxophone, trumpet, piano, upright bass, improvisation. Average energy 0.27, acoustic 0.74. Good for dinner, cafes, evenings, romantic or sophisticated settings, relaxed reading.

## genre: synthwave
Retro 1980s-style electronic music with analog synths, drum machines and neon, night-drive atmosphere. Average energy 0.74, acoustic 0.14. Good for night driving, gaming, coding, nostalgic and moody vibes.

## genre: country
Storytelling songs with guitar, fiddle and twang, about home, love, trucks and small towns. Average energy 0.59, acoustic 0.30. Good for road trips, summer, barbecues, back-roads driving.

## genre: hip hop
Rap vocals over beats: trap, boom bap and mainstream hip hop. Average energy 0.70, acoustic 0.17. Good for workouts, confidence, hype, parties, driving.

## genre: r&b
Rhythm and blues: smooth vocals, groove and soul, often about love and relationships. Average energy 0.59, acoustic 0.25. Good for romantic nights, date nights, slow dancing, late-night chill.

## genre: edm
Electronic dance music: house, EDM, dubstep, drops and four-on-the-floor beats. Average energy 0.76, acoustic 0.11. Good for parties, clubs, festivals, running, cardio, euphoric highs.

## genre: classical
Orchestral and solo piano works by composers such as Mozart, Chopin, Debussy and Vivaldi. Average energy 0.22, acoustic 0.92. Good for studying, deep focus, reading, sleep, calm and elegant moods.

## genre: reggae
Jamaican music with an offbeat skank guitar, deep bass and laid-back groove. Average energy 0.61, acoustic 0.18. Good for beach days, summer, sunshine, relaxing with friends, positive vibes.

## genre: blues
Guitar and vocal music built on blues scales and feeling, from Delta blues to electric blues. Average energy 0.54, acoustic 0.35. Good for heartbreak, rainy evenings, slow nights, soulful and sad moods.

## genre: folk
Acoustic singer-songwriter music with guitar, banjo and honest lyrics. Average energy 0.48, acoustic 0.51. Good for campfires, quiet mornings, hiking, nostalgic and reflective moods.

## mood: happy
Bright, cheerful, feel-good songs with high positivity. Average energy 0.76. Fits parties, sunny days, celebrations, good mornings.

## mood: chill
Calm, easygoing and soft songs. Average energy 0.25, acoustic 0.83. Fits studying, relaxing, winding down, lazy Sundays.

## mood: relaxed
Gentle, unhurried and peaceful. Average energy 0.33, acoustic 0.85. Fits unwinding after work, baths, slow mornings.

## mood: focused
Steady, low-distraction music that helps concentration. Average energy 0.40, acoustic 0.70. Fits studying, homework, coding, deep work, reading.

## mood: intense
Driving, powerful, high-pressure songs. Average energy 0.88. Fits workouts, gym, sprinting, competition, adrenaline.

## mood: aggressive
Angry, heavy and forceful songs. Average energy 0.82, acoustic 0.04. Fits venting, lifting weights, rage, mosh pits.

## mood: euphoric
Ecstatic, soaring peaks, built for the drop. Average energy 0.89. Fits festivals, clubs, dancing, celebrations.

## mood: confident
Swagger and self-belief, strong beats. Average energy 0.69. Fits getting ready, walking into a room, hype before a game or exam.

## mood: moody
Dark, brooding and atmospheric but not slow. Average energy 0.68. Fits night drives, gaming, rainy cities, introspection with energy.

## mood: sad
Heartbreak, loss and tears. Average energy 0.34, acoustic 0.70. Fits breakups, missing someone, crying, grief.

## mood: melancholic
Wistful, quiet sadness, often instrumental and very soft. Average energy 0.09, acoustic 0.95. Fits rainy days, late nights alone, reflection, sleep.

## mood: romantic
Love songs and tender moments. Average energy 0.45. Fits date nights, dinner, slow dancing, anniversaries, Valentine's Day.

## mood: nostalgic
Bittersweet longing for the past, throwback feeling. Average energy 0.58, acoustic 0.73. Fits road trips, old memories, autumn, looking through photos.

## mood: uplifting
Hopeful, inspiring and encouraging. Average energy 0.56. Fits mornings, motivation, fresh starts, recovering from a bad day.

## mood: hopeful
Soft optimism, often gentle piano or acoustic. Average energy 0.33, acoustic 0.95. Fits quiet mornings, healing, new beginnings.

## mood: playful
Fun, bouncy and silly. Average energy 0.88. Fits parties, dancing with friends, games. Very rare in the catalog (2 songs).

## activity: studying
For studying, homework, reading or coding, people usually want low distraction: lofi, classical, ambient or jazz, a focused or chill mood, energy around 0.3 to 0.45, and acoustic or instrumental sound.

## activity: workout
For the gym, running, lifting or cardio, people usually want high energy around 0.85 to 0.95: metal, edm, hip hop or rock with an intense, aggressive or confident mood, non-acoustic.

## activity: road trip
For road trips and long drives, people usually want upbeat sing-along music around energy 0.65 to 0.8: pop, indie pop, country or rock with a happy, nostalgic or uplifting mood.

## activity: sleep
For sleep, falling asleep, bedtime, insomnia, naps and winding down, people usually want the calmest music, energy 0.05 to 0.2: ambient or classical with a melancholic, relaxed or chill mood and very acoustic sound.

## activity: party
For parties, dancing and pregaming, people usually want energy around 0.8 to 0.9: edm, pop or hip hop with a happy, euphoric or confident mood, non-acoustic.

## activity: heartbreak
For a breakup, getting dumped, a broken heart, missing an ex or a sad night, people usually want low to medium energy around 0.3 to 0.4: blues, folk, indie pop or r&b with a sad or melancholic mood, often acoustic.

## activity: date night
For a date night or romantic dinner, people usually want energy around 0.4 to 0.5: r&b, jazz or pop with a romantic mood.

## activity: night drive
For night drives and late-night gaming, people usually want energy around 0.7 to 0.8: synthwave or electronic music with a moody or confident mood, non-acoustic.

## activity: rainy day
For rainy days, people usually want cozy low energy around 0.2 to 0.4: lofi, jazz or folk with a chill, melancholic or sad mood, acoustic.
