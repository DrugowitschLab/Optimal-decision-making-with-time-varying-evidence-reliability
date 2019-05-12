
using HDF5, JLD

# writes x to group g under name v, with compression
function writecompressed{T<:Number}(g::HDF5Group, v, x::AbstractArray{T})
    g[v, "chunk", size(x), "compress", 9] = x
end
writecompressed(g::HDF5Group, v, x) = write(g, v, x)

# writes variables vars to group gname in file fname
macro writedata(fname, gname, vars...)
    writeexprs = Array(Expr, length(vars))
    for i in 1:length(vars)
        # remove daata first, if already exists, then write in all cases
        writeexprs[i] = Expr(:block,
            :(if $(string(vars[i])) in names(g); o_delete(g, $(string(vars[i]))); end),
            :(writecompressed(g.plain, $(string(vars[i])), $(esc(vars[i])))))
    end
    # open file "r+" if already exists, "w" otherwise
    Expr(:block,
         :(local f = jldopen($fname, isfile($fname) ? "r+" : "w")),
         Expr(:try, 
            Expr(:block,
                :(local g = $gname in names(f) ? f[$gname] : g_create(f, $gname)),
                writeexprs...),
            false, false, :(close(f)))
        )
end


# reads variables vars from group gname in file fname
macro readdata(fname, gname, vars...)
    readexprs = Array(Expr, length(vars))
    for i in 1:length(vars)
        readexprs[i] = :($(esc(vars[i])) = read(g, $(string(vars[i]))))
    end
    Expr(:block,
         :(local f = jldopen($fname, "r")),
         Expr(:try, 
            Expr(:block, 
                :(local g = f[$gname]),
                Expr(:global, map(esc, vars)...),
                readexprs...),
            false, false, :(close(f)))
        )
end


# computes a histogram and binned statistics of additional variables
#
# v is a vector of variables to be binned into n bins. vars is a list of
# further variables (vectors the same size as v) whose statistics are computed
# for each bin.
#
# The function returns (e, counts, stats...), where e is the range of bin
# edges, counts is a vector over counts of bins, and stats is a sequence of
# n x 2 statistic matrices, which means and SEM per bin in the first and
# and second column, respectively.
function histbin(v, n::Integer, vars...)
    minv, maxv = minimum(v), maximum(v)
    w = (maxv - minv) / n
    nvars, nv = length(vars), length(v)
    counts = zeros(Integer, n)
    stats = [zeros(Float64, n, 2) for i = 1:nvars]
    # collect counts and <X> / <X^2> statistics
    for i = 1:nv
        const b = (v[i] - minv) / w
        const bi = b >= float(n) ? n : ifloor(b) + 1
        counts[bi] += 1
        for j = 1:nvars
            const varsji = vars[j][i]
            stats[j][bi,:] += [varsji (varsji * varsji)]
        end
    end
    # turn statistics into mean and SEM
    for i = 1:n
        const ci = counts[i]
        if ci == 0
            for j = 1:nvars
                stats[j][i,:] = fill(NaN, 2)
            end
        elseif ci == 1
            for j = 1:nvars
                stats[j][i,2] = NaN
            end
        else
            for j = 1:nvars
                stats[j][i,:] /= ci
                stats[j][i,2] = sqrt(max(0.0, stats[j][i,2] - stats[j][i,1]^2) / ci)
            end
        end
    end
    tuple(minv:w:maxv, counts, stats...)
end


