from problog.tasks.dtproblog import dtproblog
from problog.program import PrologString
from problog import get_evaluatable

model = """0.3::a.  query(a)."""
print(get_evaluatable("sdd").create_from(PrologString(model)).evaluate())
model = """
0.3::rain.
0.5::wind.

?::umbrella.
?::raincoat.

broken_umbrella :- umbrella, rain, wind.
dry :- rain, raincoat.
dry :- rain, umbrella, not broken_umbrella.
dry :- not(rain).


utility(broken_umbrella, -40).
utility(raincoat, -20).
utility(umbrella, -2).
utility(dry, 60)."""
program = PrologString(model)
decisions, score, statistics = dtproblog(program)

for name, value in decisions.items():
    print ('%s: %s' % (name, value))
