---
layout: default
---
### Authors
[Ali G. Moghaddam]

### Abstract


* * * * * *
### Discrete-time quantum trajectories

It is easy to understood that applying certain unitaries can generate entangled states from unentangled ones.
For instance, considering two qubits in either of states $$|01\rangle$$ or $$|10\rangle$$ by applying the two-qubit unitary operator

$$
{\cal U}_\alpha=
\begin{pmatrix}
1&0&0&0\\
0&\cos \alpha&\sin\alpha&0\\
0&-\sin\alpha&\cos \alpha&0\\
0&0&0&1\\
\end{pmatrix},
$$

we get entangled states $$\cos \alpha|01\rangle -\sin \alpha|10\rangle  $$ and $$\sin \alpha|01\rangle+ \cos \alpha|10\rangle $$, respectively.
For $$\alpha=\pi/4$$ these two states will be maximally entangled states (singlet and $$S=0$$ triplet states, respectively).
This idea can be exploited to design well known quantum circuits to generate on-demand highly-entangled states by successive application of  
two-qubit unitaries to a multi-qubit system shown.

Before going to the details of how a quantum circuit operates, we should introduce the useful notation of tensors for presenting manybody states.
A general manybody state of $$2N$$ $$n$$-bit chain
can be written as

$$
|\Psi\rangle =\sum_{\{i,j\}=0,1,\cdots,n-1} \psi_{i_1j_1\cdots i_Nj_N} |i_1j_1\cdots i_Nj_N \rangle. 
$$



<!--

Text can be **bold**, _italic_, or ~~strikethrough~~.

[Link to another page](another-page).

There should be whitespace between paragraphs.

There should be whitespace between paragraphs. We recommend including a README, or a file with information about your project.

# [](#header-1)Header 1

This is a normal paragraph following a header. GitHub is a code hosting platform for version control and collaboration. It lets you and others work together on projects from anywhere.

## [](#header-2)Header 2

> This is a blockquote following a header.
>
> When something is important enough, you do it even if the odds are not in your favor.

### [](#header-3)Header 3

```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version}")
end
```

#### [](#header-4)Header 4

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

##### [](#header-5)Header 5

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

###### [](#header-6)Header 6

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

### There's a horizontal rule below this.

* * *

### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip

### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four

### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item

### Small image

![](https://assets-cdn.github.com/images/icons/emoji/octocat.png)

### Large image

![](https://guides.github.com/activities/hello-world/branching.png)


### Definition lists can be used with HTML syntax.

<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
```

```
The final element.
```

-->
