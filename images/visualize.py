import numpy as np
import plotly.express as px
import plotly.graph_objects as go
# %%
input = np.loadtxt("../results/input.txt")
inputS = np.loadtxt("../results/input_spectr.txt")
output = np.loadtxt("../results/output.txt")
outputS = np.loadtxt("../results/output_spectr.txt")
output_back = np.loadtxt("../results/output_back.txt")
output_backS = np.loadtxt("../results/output_back_spectr.txt")

t = np.loadtxt("../results/time.txt")
w = np.loadtxt("../results/freq.txt")
# %%
E_start = input[::2] + 1j*input[1::2]
E_startS = inputS[::2] + 1j*inputS[1::2]
E_end = output[::2] + 1j*output[1::2]
E_endS = outputS[::2] + 1j*outputS[1::2]
E_back = output_back[::2] + 1j*output_back[1::2]
E_backS = output_backS[::2] + 1j*output_backS[1::2]

I_start = np.abs(E_start*np.conj(E_start))
I_startS = np.abs(E_startS*np.conj(E_startS))
I_end = np.abs(E_end*np.conj(E_end))
I_endS = np.abs(E_endS*np.conj(E_endS))
I_back = np.abs(E_back*np.conj(E_back))
I_backS = np.abs(E_backS*np.conj(E_backS))

W_start = np.trapz(I_start, t)
W_end = np.trapz(I_end, t)
W_back = np.trapz(I_back, t)

print("Energy input = %.2f" % W_start)
print("Energy back = %.2f" % W_back)
print("Energy output = %.2f" % W_end)
# %% Input and back propagated signal intensity
nt1 = abs(t - (-40)).argmin()
nt2 = abs(t - 50).argmin()
        
fig = go.Figure()

fig.add_trace(go.Scatter(x=t[nt1:nt2], y=I_start[nt1:nt2],
                    mode='lines',
                    name='Input'))

fig.add_trace(go.Scatter(x=t[nt1:nt2], y=I_back[nt1:nt2],
                    mode='lines',
                    name='Back'))

fig.update_layout(title_text="Signal intensity")
fig.update_xaxes(title_text="Time")
fig.update_yaxes(title_text="Intensity")

fig.write_image("input.svg")
# %% Output signal intensity
fig = px.line(x=t, y=I_end,
              title='Output signal intensity',
              labels={'x':'Intensity', 'y':'Time'})

fig.write_image("output.svg")
# %% Spectrum
fig = go.Figure()

fig.add_trace(go.Scatter(x=w, y=I_startS,
                    mode='lines',
                    name='Input'))

fig.add_trace(go.Scatter(x=w, y=I_endS,
                    mode='lines',
                    name='Output'))

fig.add_trace(go.Scatter(x=w, y=I_backS,
                    mode='lines',
                    name='Back'))

fig.update_layout(yaxis_type="log")
fig.update_layout(title_text="Spectrum intensity")
fig.update_xaxes(title_text="Frequency")
fig.update_yaxes(title_text="Intensity (log scale)")

fig.write_image("spectrum.svg")
