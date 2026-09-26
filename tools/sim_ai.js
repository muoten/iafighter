// Headless win-rate check: IA_FRA (danAI) vs a scripted player bot, using the game's own logic.
// usage: node tools/sim_ai.js [game/index.html] [rounds] [demo]   (demo = the page's own Congui bot)
const fs = require('fs');
const html = fs.readFileSync(process.argv[2] || 'game/index.html', 'utf8');
const js = html.slice(html.indexOf('// ── fighters ──'), html.indexOf('// ── loop ──'));
const window = {}, document = { getElementById: () => ({}) }, C = {}, location = { hash:'' }, NAME = { joe:'CONGUI', dan:'IA_FRA' }, W = 320, GROUND = 226, soundOn = false, music = { pause(){}, play(){ return Promise.resolve(); } };
let seed = 7; Math.random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
eval(js.replace(/^let joe, dan/m, 'var joe, dan'));
function bot(me, o){                                                         // a middling human: walks in, mashes in range
  const d = Math.abs(o.x - me.x), inp = { left:false, right:false, down:false, upE:false, punchE:false, kickE:false };
  const toward = o.x > me.x ? 'right' : 'left';
  if (d > 44) inp[toward] = true; else if (Math.random() < .12) inp[Math.random() < .5 ? 'punchE' : 'kickE'] = true;
  return inp;
}
const used = { punch:0, kick:0, spin:0 };
let wins = 0, N = +(process.argv[3] || 400), hpLeft = 0, joeLeft = 0;
for (let r = 0; r < N; r++) {
  joe = fighter('joe', 90, 1); dan = fighter('dan', 230, -1); round = 1; newRound(); state = 'fight'; ai = { t:0, want:'idle' };
  for (let t = 0; t < 99 * 60 && state === 'fight'; t++) {
    control(joe, dan, process.argv[4] === 'demo' ? congBot() : bot(joe, dan)); const was = dan.state; control(dan, joe, danAI()); if (dan.state === 'attack' && was !== 'attack') used[dan.atk]++; physics(joe); physics(dan);
    if (Math.abs(joe.x - dan.x) < 26) { const p = (26 - Math.abs(joe.x - dan.x)) / 2 * (joe.x < dan.x ? 1 : -1); joe.x -= p; dan.x += p; }
  }
  if (dan.hp > joe.hp) wins++; hpLeft += dan.hp; joeLeft += joe.hp;
}
console.log(`IA_FRA punches ${used.punch} / kicks ${used.kick} / spins ${used.spin} · wins ${(100 * wins / N).toFixed(0)}% of ${N} rounds, avg HP left IA_FRA ${(hpLeft / N).toFixed(0)} vs player ${(joeLeft / N).toFixed(0)}`);
