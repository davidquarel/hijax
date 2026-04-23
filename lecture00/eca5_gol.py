# %%
"""
Lecture 00: Hi, JAX! How's life?

Demonstration: Port an elementary cellular automaton simulator from NumPy to
JAX.

Learning objectives:

1. Introduction to jax.numpy (jnp)
2. First look at some JAX function transformations (jit, vmap)
3. Exploration of performance aspects of NumPy, JAX, JIT
"""

import time
import numpy as np
import jax
import jax.numpy as jnp

import tyro
import einops
from jaxtyping import Array, UInt8, Bool
import matthewplotlib as mp
from PIL import Image


"""
8 neighbours (moore neighbourhood)
ALIVE -> ALIVE (2 or 3 adjacent)
      -> DEAD (else)
      
DEAD -> ALIVE (3 adj.)
     -> DEAD (else)
"""

#gpt slop
def play_video(frames, fps=12):
    delay = 1.0 / fps

    # print a newline to ensure we're not mid-line
    print()

    # mark start position
    print("\x1b[s", end="")  # save cursor

    for frame in frames:
        time.sleep(delay)

        # restore to exact saved position
        print("\x1b[u", end="")

        p = mp.image(frame)

        # IMPORTANT: ensure full overwrite
        s = p.renderstr()

        # force newline padding to avoid wrap issues
        if not s.endswith("\n"):
            s += "\n"

        print(s, end="", flush=True)
# end gpt slop

def main(
    size: int = 10,
    num_steps: int = 50,
    alive_frac: float = 0.2,
    seed: int = 0,
    print_image: bool = True,
    save_image: bool = False,
):
    start_time = time.perf_counter()
    
    compiled_simulate = jax.jit(
        simulate, 
        static_argnames=['size', 'num_steps']
    )

    key = jax.random.PRNGKey(seed)
    init_state = jax.random.bernoulli(key, p=alive_frac, shape=(size, size)).astype(jnp.uint8)

    
    states = compiled_simulate(
        init_state,
        size,
        num_steps,
    ).block_until_ready()
    end_time = time.perf_counter()
    print("compilation complete!")
    print(f"time taken {end_time - start_time:.5f} seconds")
    
    
    start_time = time.perf_counter()
    states_tyx = compiled_simulate(
        init_state,
        size,
        num_steps,
    ).block_until_ready()
    end_time = time.perf_counter()
    
    print("simulation complete!")
    print(f"time taken {end_time - start_time:.5f} seconds")

    if print_image:
        play_video(states_tyx*255, fps=30)

    # if save_image:
    #     print("rendering to 'output.png'...")
       
    #     numpy_state = np.asarray(states_tyx)
    #     mp
    #     # Image.fromarray(255 - 255 * numpy_state).save('output.png')


def simulate(
    init_state: UInt8[Array, "size size"],
    size: int,
    num_steps: int,
) -> UInt8[Array, "num_steps size size"]:
    
    # next_alive = alive ? 2 <= n <= 3 : n == 2

    def step(alive, _) -> tuple: 
        state_wrapped : UInt8[Array, "width+2, width+2"]
        state_wrapped = jnp.pad(alive, 1, mode='wrap') 
        
        north_west = state_wrapped[:-2,  :-2]
        north      = state_wrapped[:-2, 1:-1]
        north_east = state_wrapped[:-2, 2:  ]
        
        west       = state_wrapped[1:-1,  :-2]
        #alive      = state_wrapped[1:-1, 1:-1]
        east       = state_wrapped[1:-1, 2:  ]
        
        south_west = state_wrapped[2:,  :-2]
        south      = state_wrapped[2:, 1:-1]
        south_east = state_wrapped[2:, 2:  ]
        
        neighbours = north_west + north + north_east + west + east + south_west + south + south_east
        
        istwo = neighbours == 2
        isthree = neighbours == 3
        
        next_state = (alive & (istwo | isthree)) | (~alive & isthree) 
        
        return next_state, next_state
     
        
    _, next_states = jax.lax.scan(
        step,
        init_state,
        length=num_steps-1
    )

    return jnp.concatenate(
        [init_state[None], next_states],
        axis=0
    )
    

if __name__ == "__main__":
    tyro.cli(main)

# %%
